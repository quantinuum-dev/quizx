use crate::graph::*;
use crate::phase::Phase;
use crate::scalar::*;
use num::complex::ComplexFloat;
use num::rational::Ratio;
use num::Complex;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;

use super::SimpFunc;

/// Store the (partial) decomposition of a graph into stabilisers
#[derive(Clone)]
pub struct ApproxDecomposer {
    simp_func: SimpFunc,
}

type PickDecomposition<G> = dyn Fn(G) -> G;

pub trait DecomposeFn {
    fn decompose<'a, G: GraphLike>(
        &'a self,
        graph: &'a G,
    ) -> Vec<(ScalarN, Box<PickDecomposition<G>>)>;
}

impl ApproxDecomposer {
    pub fn run<G: GraphLike>(
        &self,
        graph: &G,
        iters: usize,
        decomposer: &impl DecomposeFn,
    ) -> Complex<f64> {
        let mut scalar = ScalarN::zero();
        for _ in 0..iters {
            scalar += self.run_one(graph, decomposer);
        }
        let s = scalar.complex_value();
        Complex::new(s.re / iters as f64, s.im / iters as f64)
    }

    fn run_one<G: GraphLike>(&self, graph: &G, decomposer: &impl DecomposeFn) -> ScalarN {
        let mut graph = graph.clone();
        while graph.tcount() > 0 {
            let options = decomposer.decompose(&graph);
            let choice: Box<PickDecomposition<G>> = self.pick(options);
            graph = choice(graph);
            match self.simp_func {
                SimpFunc::FullSimp => {
                    crate::simplify::full_simp(&mut graph);
                }
                SimpFunc::CliffordSimp => {
                    crate::simplify::clifford_simp(&mut graph);
                }
                _ => {}
            }
        }

        // No T-s left, graph should be fully reduced
        if graph.num_vertices() != 0 {
            panic!("Graph was not fully reduced");
        }

        graph.scalar().clone()
    }

    /// Given a list of decomposition options weighted by some scalar, pick one
    /// randomly based on the absolute value of the scalar.
    fn pick<'a, G: GraphLike>(
        &self,
        options: Vec<(ScalarN, Box<PickDecomposition<G>>)>,
    ) -> Box<PickDecomposition<G>> {
        // Convert the scalars to their absolute values, so we can use them as weights when picking
        let options = options
            .into_iter()
            .map(|(s, f)| (s.complex_value().abs(), f));
        // Collect both values
        let (weights, mut options): (Vec<_>, Vec<_>) = options.into_iter().unzip();
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        options.swap_remove(dist.sample(&mut rng))
    }
}

struct DumbTDecomposer;

impl DecomposeFn for DumbTDecomposer {
    fn decompose<'a, G: GraphLike>(
        &'a self,
        graph: &'a G,
    ) -> Vec<(ScalarN, Box<PickDecomposition<G>>)> {
        // Find the first T spider
        let v = graph
            .vertices()
            .into_iter()
            .find(|v| graph.phase(*v).is_t())
            .unwrap();

        let id_case = move |mut g: G| -> G {
            g.set_phase(v, Phase::zero());
            g
        };
        let s_case = move |mut g: G| -> G {
            g.set_phase(v, Ratio::new(1, 2));
            *g.scalar_mut() *= ScalarN::from_phase(Ratio::new(-1, 4));
            g
        };

        let mut res: Vec<(ScalarN, Box<PickDecomposition<G>>)> = vec![];
        res.push((ScalarN::one(), Box::new(id_case)));
        res.push((ScalarN::one(), Box::new(s_case)));
        res
    }
}
