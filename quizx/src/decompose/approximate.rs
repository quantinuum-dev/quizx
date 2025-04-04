use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use crate::circuit::Circuit;
use crate::graph::*;
use crate::phase::Phase;
use crate::scalar::*;
use crate::simplify::clifford_simp;
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
    parallel: bool,
}

type PickDecomposition<G> = dyn Fn(&mut G);

pub trait DecomposeFn: Sync {
    fn decompose<'a, G: GraphLike>(
        &'a self,
        graph: &'a G,
    ) -> Vec<(ScalarN, Box<PickDecomposition<G>>)>;

    fn required_iters(&self, tcount: usize, eps: f64) -> usize;
}

impl ApproxDecomposer {
    pub fn new(simp_func: SimpFunc, parallel: bool) -> ApproxDecomposer {
        ApproxDecomposer {
            simp_func,
            parallel,
        }
    }

    pub fn run<G: GraphLike, D: DecomposeFn>(
        &self,
        graph: &G,
        eps: f64,
        decomposer: &D,
    ) -> Complex<f64> {
        if self.parallel {
            self.run_serial(graph, eps, decomposer)
        } else {
            self.run_parallel(graph, eps, decomposer)
        }
    }

    fn run_serial<G: GraphLike, D: DecomposeFn>(
        &self,
        graph: &G,
        eps: f64,
        decomposer: &D,
    ) -> Complex<f64> {
        let mut required_iters = decomposer.required_iters(graph.tcount(), eps);
        let mut total_iters = 0;
        let mut scalar = ScalarN::zero();

        while required_iters > 0 {
            let (s, iter_reduction) = self.run_one(graph, eps, decomposer);
            required_iters = required_iters.saturating_sub(iter_reduction);
            scalar += s;
            total_iters += 1;
        }

        scalar.complex_value()
            * ((decomposer.required_iters(graph.tcount(), 1.0) as f64).sqrt() / total_iters as f64)
    }

    fn run_parallel<G: GraphLike, D: DecomposeFn>(
        &self,
        graph: &G,
        eps: f64,
        decomposer: &D,
    ) -> Complex<f64> {
        let max_iters = decomposer.required_iters(graph.tcount(), eps);
        let total = AtomicUsize::new(0);
        let remaining_iters = AtomicUsize::new(max_iters);

        let scalar = (0..max_iters)
            .into_par_iter()
            .map(|_| self.run_one(graph, eps, decomposer))
            .take_any_while(|(_, reduction)| {
                // println!("{}", reduction);
                total.fetch_add(1, Relaxed);
                remaining_iters
                    .fetch_update(Relaxed, Relaxed, |q| q.checked_sub(*reduction))
                    .is_ok()
            })
            .map(|(s, _)| s)
            .reduce(ScalarN::zero, |s1, s2| s1 + s2);

        println!("max: {}, actual: {}", max_iters, total.load(Relaxed));
        scalar.complex_value()
            * ((decomposer.required_iters(graph.tcount(), 1.0) as f64).sqrt()
                / total.load(Relaxed) as f64)
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
        let c = self.run(&g, eps, decomposer);
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
        for step in 0..mixing_steps {
            println!("{}", step);
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

    fn run_one<G: GraphLike, D: DecomposeFn>(
        &self,
        graph: &G,
        eps: f64,
        decomposer: &D,
    ) -> (ScalarN, usize) {
        let mut graph = graph.clone();
        let initial_tcount = graph.tcount();
        let mut curr_tcount = initial_tcount;
        let mut iter_reduction = 0;
        let mut depth = 0;
        while curr_tcount > 0 {
            let options = decomposer.decompose(&graph);
            let choice: Box<PickDecomposition<G>> = self.pick(options);
            choice(&mut graph);
            // curr_tcount -= 1;
            self.simplify(&mut graph);
            depth += 1;
            // Check how many Ts where cancelled
            let old_tcount = curr_tcount;
            curr_tcount = graph.tcount();
            let mut num_cancelled = old_tcount - curr_tcount - 1;
            // No need to decompose further if we produced a zero scalar
            if graph.scalar().is_zero() {
                num_cancelled = old_tcount;
            }
            // Compute how many samples those cancelled T gates save
            let mut saved_iters = (0..num_cancelled)
                .map(|i| decomposer.required_iters(curr_tcount + i, eps))
                .fold(0, |a, b| a + b);
            // Divide by number of expected samples that will reach
            let exp_visits =
                decomposer.required_iters(initial_tcount, eps) as f64 / 2.0.powi(depth);
            if exp_visits > 1.0 {
                saved_iters = (saved_iters as f64 / exp_visits) as usize;
            }
            iter_reduction += saved_iters;
            // No need to decompose further if we produced a zero scalar
            if graph.scalar().is_zero() {
                return (ScalarN::zero(), iter_reduction + 1);
            }
        }

        // No T-s left, graph should be fully reduceable
        clifford_simp(&mut graph);
        if graph.num_vertices() != 0 {
            panic!("Graph was not fully reduced");
        }
        // println!("{}", graph.scalar());

        (graph.scalar().clone(), iter_reduction + 1)
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
            .find(|v| !graph.phase(*v).is_clifford())
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

    fn required_iters(&self, tcount: usize, eps: f64) -> usize {
        (2.0f64.powf(0.23 * tcount as f64) / (eps * eps)) as usize
    }
}
