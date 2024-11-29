use crate::circuit::Circuit;
use crate::graph::{self, *};
use crate::phase::Phase;
use crate::scalar::*;
use crate::vec_graph::Graph;
use itertools::Itertools;
use num::complex::ComplexFloat;
use num::rational::Ratio;
use num::Complex;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use super::SimpFunc;

/// Store the (partial) decomposition of a graph into stabilisers
#[derive(Clone)]
pub struct ApproxDecomposer {
    simp_func: SimpFunc,
}

type PickDecomposition<G> = dyn Fn(&mut G);

pub trait DecomposeFn: Sync {
    fn decompose<'a, G: GraphLike>(
        &'a self,
        graph: &'a G,
    ) -> Vec<(ScalarN, Box<PickDecomposition<G>>)>;
}

impl ApproxDecomposer {
    pub fn new(simp_func: SimpFunc) -> ApproxDecomposer {
        ApproxDecomposer { simp_func }
    }

    pub fn run<G: GraphLike, D: DecomposeFn>(
        &self,
        graph: &G,
        iters: usize,
        decomposer: &D,
    ) -> Complex<f64> {
        let scalar = (0..iters)
            .into_par_iter()
            .map(|_| self.run_one(graph, decomposer))
            .reduce_with(|s1, s2| s1 + s2)
            .unwrap();
        let s = scalar.complex_value();
        Complex::new(s.re / iters as f64, s.im / iters as f64)
    }

    pub fn amplitude<D: DecomposeFn>(
        &self,
        circ: &Circuit,
        eps: f64,
        xs: &[bool],
        decomposer: &D,
    ) -> f64 {
        let mut g: Graph = circ.to_graph();
        g.plug_inputs(&vec![BasisElem::Z0; circ.num_qubits()]);
        g.plug_outputs(
            &xs.iter()
                .map(|x| if *x { BasisElem::Z0 } else { BasisElem::Z1 })
                .collect_vec(),
        );
        self.simplify(&mut g);
        let iters = 1;
        let c = self.run(&g, iters, decomposer);
        (c * c.conj()).re()
    }

    pub fn metropolis_sample<D: DecomposeFn>(
        &self,
        circ: &Circuit,
        mixing_steps: usize,
        eps: f64,
        decomposer: &D,
    ) -> Vec<bool> {
        let mut rng = thread_rng();
        let n = circ.num_qubits();
        let mut xs = (0..n).map(|_| rng.gen()).collect_vec();
        let mut p = self.amplitude(circ, eps, &xs, decomposer);
        for _ in 0..mixing_steps {
            let i = rng.gen_range(0..n);
            let mut xs_new = xs.clone();
            xs_new[i] = !xs[i];
            let p_new = self.amplitude(circ, eps, &xs, decomposer);
            if p_new > p || (p > 0.0 && rng.gen_bool(p_new / p)) {
                xs = xs_new;
                p = p_new;
            }
        }
        xs
    }

    fn run_one<G: GraphLike, D: DecomposeFn>(&self, graph: &G, decomposer: &D) -> ScalarN {
        let mut graph = graph.clone();
        while graph.tcount() > 0 {
            let options = decomposer.decompose(&graph);
            let choice: Box<PickDecomposition<G>> = self.pick(options);
            choice(&mut graph);
            self.simplify(&mut graph);
            // No need to decompose further if we produced a zero scalar
            if graph.scalar().is_zero() {
                return ScalarN::zero();
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

    fn simplify<G: GraphLike>(&self, graph: &mut G) {
        match self.simp_func {
            SimpFunc::FullSimp => {
                crate::simplify::full_simp(graph);
            }
            SimpFunc::CliffordSimp => {
                crate::simplify::clifford_simp(graph);
            }
            _ => {}
        }
    }
}

pub struct DumbTDecomposer;

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

        let id_case = move |g: &mut G| {
            g.set_phase(v, Phase::zero());
        };
        let s_case = move |g: &mut G| {
            g.set_phase(v, Ratio::new(1, 2));
            *g.scalar_mut() *= ScalarN::from_phase(Ratio::new(-1, 4));
        };

        let mut res: Vec<(ScalarN, Box<PickDecomposition<G>>)> = vec![];
        res.push((ScalarN::one(), Box::new(id_case)));
        res.push((ScalarN::one(), Box::new(s_case)));
        res
    }
}
