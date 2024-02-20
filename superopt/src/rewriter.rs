//! Rewriter for the SuperOptimizer.

use std::collections::{HashMap, HashSet};
use std::ops::Index;

use itertools::Itertools;
use quizx::vec_graph::{EType, VType};
use quizx::{
    flow::causal::{CausalFlow, ConvexHull},
    graph::GraphLike,
    portmatching::{CausalMatcher, CausalPattern, PatternID},
    vec_graph::V,
};

use crate::rewrite_sets::RuleSide;
use crate::{
    cost::CostDelta,
    rewrite_sets::{RewriteRhs, RewriteSet},
};

pub trait Rewriter {
    type Rewrite;

    /// Get the rewrites that can be applied to the graph.
    fn get_rewrites(&self, graph: &impl GraphLike) -> Vec<Self::Rewrite>;

    /// Apply the rewrites to the graph.
    fn apply_rewrite<G: GraphLike>(&self, rewrite: Self::Rewrite, graph: &G) -> RewriteResult<G>;
}

pub struct RewriteResult<G> {
    pub graph: G,
    pub cost_delta: CostDelta,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RhsIdx(usize);

/// A rewriter that applies causal flow preserving rewrites.
///
/// The set of possible rewrite rule are given as a list of `RewriteSet`s.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CausalRewriter<G: GraphLike> {
    matcher: CausalMatcher<G>,
    lhs_to_rhs: HashMap<PatternID, RhsIdx>,
    all_rhs: Vec<Vec<CausalRewriterRHS>>,
}

#[derive(Clone, Debug)]
pub struct Rewrite {
    /// The nodes matching the LHS boundary in the matched graph.
    lhs_boundary: Vec<V>,
    /// The internal nodes of the LHS in the matched graph.
    lhs_internal: HashSet<V>,
    /// Number of new internal nodes in the RHS.
    rhs_n_internal: usize,
    /// Edges to add to the graph.
    rhs_new_edges: Vec<(RHSVertex, RHSVertex)>,
    /// The cost delta of the rewrite.
    ///
    /// Negative delta is an improvement (cost decrease).
    cost_delta: CostDelta,
    /// The pattern ID of the matched LHS
    pattern_id: PatternID,
}

impl Rewrite {
    fn lhs_vertices(&self) -> impl Iterator<Item = &V> + '_ {
        self.lhs_boundary.iter().chain(&self.lhs_internal)
    }

    /// Whether the rewrite is flow preserving when applied on `graph`.
    ///
    /// TODO: This can be done faster by pre-computing a "causal structure"
    fn is_flow_preserving(&self, graph: &impl GraphLike, flow: &CausalFlow) -> bool {
        let hull = ConvexHull::from_region(self.lhs_vertices(), graph, flow);
        let (mut subgraph, vmap) = graph.induced_subgraph(hull.vertices());
        subgraph.set_inputs(hull.inputs().iter().map(|v| vmap[v]).collect());
        subgraph.set_outputs(hull.outputs().iter().map(|v| vmap[v]).collect());
        let subgraph_rw = self.map_lhs(&vmap);
        subgraph_rw.apply(&mut subgraph);
        CausalFlow::from_graph(&subgraph).is_ok()
    }

    fn map_lhs<Map>(&self, vmap: &Map) -> Self
    where
        Map: for<'a> Index<&'a V, Output = V>,
    {
        let lhs_boundary = self.lhs_boundary.iter().map(|v| vmap[v]).collect();
        let lhs_internal = self.lhs_internal.iter().map(|v| vmap[v]).collect();
        Self {
            lhs_boundary,
            lhs_internal,
            ..self.clone()
        }
    }

    fn apply<G: GraphLike>(&self, g: &mut G) -> CostDelta {
        // Set mapping from RHSVertex indices to `g`
        let mut new_r_names: HashMap<_, _> = self
            .lhs_boundary
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (RHSVertex::Boundary(i), v))
            .collect();

        // Remove the internal nodes of the LHS.
        for &v in &self.lhs_internal {
            g.remove_vertex(v);
        }

        // Insert new internal nodes for the RHS.
        new_r_names.extend(
            (0..self.rhs_n_internal).map(|i| (RHSVertex::Internal(i), g.add_vertex(VType::Z))),
        );

        // TODO: changes in phases/vtype on boundary

        // Add/Remove new edges.
        for (u, v) in &self.rhs_new_edges {
            let u = new_r_names[u];
            let v = new_r_names[v];
            // This will remove edge if it exists
            g.add_edge_smart(u, v, EType::H);
        }

        self.cost_delta
    }
}

impl<G: GraphLike> Rewriter for CausalRewriter<G> {
    type Rewrite = Rewrite;

    fn get_rewrites(&self, graph: &impl GraphLike) -> Vec<Self::Rewrite> {
        let flow = CausalFlow::from_graph(graph).expect("no causal flow");
        let mut rewrites = self
            .matcher
            .find_matches(graph, &flow)
            .flat_map(|m| {
                self.get_rhs(m.pattern_id).iter().map(move |rhs| {
                    let lhs_boundary = m.boundary.clone();
                    let lhs_internal = m.internal.clone();
                    let rhs_new_edges = rhs.edges.clone();
                    Rewrite {
                        lhs_boundary,
                        lhs_internal,
                        rhs_new_edges,
                        rhs_n_internal: rhs.n_internal,
                        cost_delta: rhs.cost_delta,
                        pattern_id: m.pattern_id,
                    }
                })
            })
            .filter(|rw| rw.is_flow_preserving(graph, &flow))
            .collect_vec();
        rewrites.sort_by_key(|rw| rw.cost_delta);
        rewrites
    }

    fn apply_rewrite<H: GraphLike>(&self, rewrite: Self::Rewrite, graph: &H) -> RewriteResult<H> {
        let mut graph = graph.clone();
        let p = self.matcher.get_pattern(rewrite.pattern_id);
        let cost_delta = rewrite.apply(&mut graph);
        RewriteResult { graph, cost_delta }
    }
}

/// Get edges in `rule_side` as RHSVertices.
fn get_edges<G: GraphLike>(
    rule_side: &impl RuleSide<G>,
    include_internal: bool,
) -> HashSet<(RHSVertex, RHSVertex)> {
    let boundary = rule_side
        .boundary()
        .enumerate()
        .map(|(i, v)| (v, RHSVertex::Boundary(i)));
    let internal = rule_side
        .internal()
        .enumerate()
        .map(|(i, v)| (v, RHSVertex::Internal(i)));
    let vertex_map: HashMap<_, _> = if include_internal {
        boundary.chain(internal).collect()
    } else {
        boundary.collect()
    };
    rule_side
        .graph()
        .edges()
        .filter_map(move |(s, t, etype)| {
            assert_eq!(etype, EType::H);
            Some((*vertex_map.get(&s)?, *vertex_map.get(&t)?))
        })
        .collect()
}

impl<G: GraphLike + Clone> CausalRewriter<G> {
    fn get_rhs(&self, lhs_idx: PatternID) -> &[CausalRewriterRHS] {
        let idx = &self.lhs_to_rhs[&lhs_idx];
        &self.all_rhs[idx.0]
    }

    pub fn from_rewrite_rules(rules: impl IntoIterator<Item = RewriteSet<G>>) -> Self {
        let mut patterns = Vec::new();
        let mut map_to_rhs = HashMap::new();
        let mut all_rhs = Vec::new();
        for rw_set in rules {
            let lhs = rw_set.lhs();
            let lhs_edges = get_edges(&lhs, false);
            let lhs_boundary_len = lhs.boundary().count();
            for rhs in rw_set.rhss() {
                assert_eq!(lhs_boundary_len, rhs.boundary().count());
            }

            let rhs_idx = RhsIdx(all_rhs.len());
            all_rhs.push(
                rw_set
                    .rhss()
                    .iter()
                    .map(|rhs| CausalRewriterRHS::from_rhs(rhs, &lhs_edges))
                    .collect(),
            );
            for (inputs, outputs) in lhs.ios() {
                let inputs = HashSet::from_iter(inputs);
                let outputs = HashSet::from_iter(outputs);
                patterns.push(CausalPattern::new(
                    lhs.graph(),
                    lhs.boundary().collect(),
                    inputs,
                    outputs,
                ));
                map_to_rhs.insert(PatternID(patterns.len() - 1), rhs_idx);
            }
        }
        CausalRewriter {
            matcher: CausalMatcher::from_patterns(patterns),
            lhs_to_rhs: map_to_rhs,
            all_rhs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
enum RHSVertex {
    Boundary(usize),
    Internal(usize),
}

/// A RHS of a causal rewrite rule.
///
/// The RHS graph is stored as the symmetric difference of edges between the LHS
/// and RHS.
#[derive(serde::Serialize, serde::Deserialize)]
struct CausalRewriterRHS {
    n_internal: usize,
    edges: Vec<(RHSVertex, RHSVertex)>,
    cost_delta: CostDelta,
}

impl CausalRewriterRHS {
    fn from_rhs<G: GraphLike>(
        rhs: &RewriteRhs<G>,
        lhs_boundary_edges: &HashSet<(RHSVertex, RHSVertex)>,
    ) -> Self {
        let rhs_edges = get_edges(rhs, true);
        let edges = rhs_edges
            .symmetric_difference(lhs_boundary_edges)
            .copied()
            .collect();
        CausalRewriterRHS {
            n_internal: rhs.internal().count(),
            edges,
            cost_delta: -rhs.reduction,
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::cost::{CostMetric, TwoQubitGateCount};
    use crate::rewrite_sets::test::rewrite_set_2qb_lc;

    use super::RHSVertex::Boundary;
    use super::*;
    use quizx::json::decode_graph;
    use quizx::vec_graph::Graph;
    use rstest::{fixture, rstest};

    /// Makes a simple graph.
    ///
    /// The graph is:
    /// ```text
    /// 0 - 8 - 2 - 3 - 6
    ///        / \
    /// 1 --- 4 - 5 - 7
    /// ```
    ///
    /// with inputs 0, 1 and outputs 6, 7.
    #[fixture]
    pub(crate) fn small_graph() -> Graph {
        let mut g = Graph::new();
        let vs = vec![
            g.add_vertex(VType::B),
            g.add_vertex(VType::B),
            g.add_vertex(VType::Z),
            g.add_vertex(VType::Z),
            g.add_vertex(VType::Z),
            g.add_vertex(VType::Z),
            g.add_vertex(VType::B),
            g.add_vertex(VType::B),
            g.add_vertex(VType::Z),
        ];

        g.set_inputs(vec![vs[0], vs[1]]);
        g.set_outputs(vec![vs[6], vs[7]]);

        g.add_edge_with_type(vs[0], vs[8], EType::N);
        g.add_edge_with_type(vs[1], vs[4], EType::N);

        g.add_edge_with_type(vs[8], vs[2], EType::H);
        g.add_edge_with_type(vs[2], vs[3], EType::H);
        g.add_edge_with_type(vs[2], vs[4], EType::H);
        g.add_edge_with_type(vs[2], vs[5], EType::H);
        g.add_edge_with_type(vs[4], vs[5], EType::H);

        g.add_edge_with_type(vs[3], vs[6], EType::N);
        g.add_edge_with_type(vs[5], vs[7], EType::N);

        g
    }

    #[fixture]
    pub(crate) fn json_simple_graph() -> Graph {
        const SIMPLE_GRAPH_JSON: &str = include_str!("../../test_files/simple-graph.json");
        decode_graph(SIMPLE_GRAPH_JSON).unwrap()
    }

    #[fixture]
    pub(crate) fn compiled_rewriter() -> CausalRewriter<Graph> {
        let rw_set = rewrite_set_2qb_lc();
        CausalRewriter::from_rewrite_rules(rw_set)
    }

    #[fixture]
    pub(crate) fn pre_compiled_rewriter() -> CausalRewriter<Graph> {
        const REWRITE_2QB_LC: &[u8] = include_bytes!("../../test_files/rewrites-2qb-lc.rwr");
        rmp_serde::from_slice(REWRITE_2QB_LC).unwrap()
    }

    #[rstest]
    #[case::small_compiled(small_graph(), compiled_rewriter())]
    #[case::json_compiled(json_simple_graph(), compiled_rewriter())]
    #[case::small_precompiled(small_graph(), pre_compiled_rewriter())]
    #[case::json_precompiled(json_simple_graph(), pre_compiled_rewriter())]
    fn test_match_apply(
        #[case] graph: Graph,
        #[case] rewriter: CausalRewriter<Graph>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cost_metric = TwoQubitGateCount::new();
        let graph_cost = cost_metric.cost(&graph);

        let rewrites = rewriter.get_rewrites(&graph);

        println!("Orig cost {graph_cost}");
        for rw in rewrites {
            let r = rewriter.apply_rewrite(rw, &graph);
            let new_cost = cost_metric.cost(&r.graph);

            println!("New cost {new_cost}");
            assert_eq!(graph_cost.saturating_add_signed(r.cost_delta), new_cost);
        }

        Ok(())
    }

    /// Test the following rewrite
    /// ```text
    ///       2 - 3         2 - 3
    ///      / \       ->
    ///     4 - 5           4 - 5
    /// ```
    #[rstest]
    fn flow_preservation(mut small_graph: Graph) {
        let rw = Rewrite {
            lhs_boundary: vec![2, 3, 4, 5],
            lhs_internal: HashSet::new(),
            rhs_n_internal: 0,
            rhs_new_edges: vec![
                (RHSVertex::Boundary(0), RHSVertex::Boundary(2)),
                (RHSVertex::Boundary(0), RHSVertex::Boundary(3)),
            ],
            cost_delta: 0,
            pattern_id: PatternID(0),
        };
        let flow = CausalFlow::from_graph(&small_graph).expect("no causal flow");
        assert!(rw.is_flow_preserving(&small_graph, &flow));

        // Apply rewrite
        rw.apply(&mut small_graph);
        assert_eq!(small_graph.num_edges(), 7);
    }

    /// Test the following rewrite
    /// ```text
    ///    0 - 1 - 2         0   2
    ///    |   |   |    ->    \ /
    ///    |   |   |    ->    / \
    ///    3 - 4 - 5         3   5
    /// ```
    #[test]
    fn rewrite() {
        let mut g = {
            let mut g = Graph::new();
            let line1 = (0..3).map(|_| g.add_vertex(VType::Z)).collect_vec();
            let line2 = (0..3).map(|_| g.add_vertex(VType::Z)).collect_vec();
            for i in 0..3 {
                if i + 1 < 3 {
                    g.add_edge_with_type(line1[i], line1[i + 1], EType::H);
                    g.add_edge_with_type(line2[i], line2[i + 1], EType::H);
                }
                g.add_edge_with_type(line1[i], line2[i], EType::H);
            }
            g
        };
        let rw = Rewrite {
            lhs_boundary: vec![0, 2, 3, 5],
            lhs_internal: [1, 4].into_iter().collect(),
            rhs_n_internal: 0,
            rhs_new_edges: vec![
                (Boundary(0), Boundary(2)),
                (Boundary(1), Boundary(3)),
                (Boundary(0), Boundary(3)),
                (Boundary(1), Boundary(2)),
            ],
            // these don't matter
            cost_delta: 0,
            pattern_id: PatternID(0),
        };
        rw.apply(&mut g);
        assert_eq!(g.num_vertices(), 4);
        assert_eq!(g.num_edges(), 2);
        assert_eq!(g.neighbors(0).collect_vec(), vec![5]);
        assert_eq!(g.neighbors(3).collect_vec(), vec![2]);
    }
}
