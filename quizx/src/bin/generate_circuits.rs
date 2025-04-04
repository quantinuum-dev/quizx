/// Generates random benchmarking circuits
use std::env;
use std::path::Path;

use quizx::circuit::Circuit;
use quizx::graph::{BasisElem, GraphLike, VType};
use quizx::json::write_graph;
use quizx::simplify::full_simp;
use quizx::vec_graph::Graph;

fn main() {
    let args: Vec<_> = env::args().collect();
    let (qs, min_ts, max_ts, max_circs, mut seed) = if args.len() >= 6 {
        (
            args[1].parse().unwrap(),
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
        )
    } else {
        (50, 10, 100, 10, 1337)
    };

    let mut num_found = vec![0; 2 * max_ts];
    for depth in min_ts..max_ts {
        for _ in 0..10 {
            let circ = Circuit::random_pauli_gadget()
                .qubits(qs)
                .depth(depth)
                .seed(seed)
                .min_weight(2)
                .max_weight(4)
                .build();

            let mut observable = Graph::new();
            for _ in 0..qs {
                let inp = observable.add_vertex(VType::B);
                let z = observable.add_vertex_with_phase(VType::Z, 1);
                let out = observable.add_vertex(VType::B);
                observable.inputs_mut().push(inp);
                observable.outputs_mut().push(out);
                observable.add_edge(inp, z);
                observable.add_edge(z, out);
            }

            let mut g: Graph = circ.to_graph();
            g.plug_inputs(&vec![BasisElem::Z0; qs]);

            let g_adj = &g.to_adjoint();
            g.plug(&observable);
            g.plug(g_adj);

            full_simp(&mut g);
            let tcount = g.tcount();

            if tcount >= min_ts && tcount <= max_ts && num_found[tcount] < max_circs {
                let path = format!("circuits/random/{tcount}_{}.json", num_found[tcount]);
                write_graph(&g, Path::new(&path)).unwrap();
            }
            num_found[tcount] += 1;
            seed += 1;
        }
    }
}
