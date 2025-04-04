// QuiZX - Rust library for quantum circuit rewriting and optimisation
//         using the ZX-calculus
// Copyright (C) 2021 - Aleks Kissinger
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;

// use std::io;
// use std::io::Write;
use quizx::circuit::*;
use quizx::decompose::approximate::{ApproxDecomposer, DumbTDecomposer};
use quizx::graph::*;
// use quizx::scalar::*;
// use quizx::tensor::*;
use quizx::decompose::{Decomposer, SimpFunc};
use quizx::vec_graph::Graph;
// use rayon::prelude::*;
// use rand::SeedableRng;
// use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let debug = true;
    let args: Vec<_> = env::args().collect();
    let (qs, depth, min_weight, max_weight, seed) = if args.len() >= 6 {
        (
            args[1].parse().unwrap(),
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
        )
    } else {
        // (50, 50, 2, 4, 1339)
        (13, 15, 2, 4, 1338)
    };
    if debug {
        println!(
            "qubits: {}, depth: {}, min_weight: {}, max_weight: {}, seed: {}",
            qs, depth, min_weight, max_weight, seed
        );
    }
    let c = Circuit::random_pauli_gadget()
        .qubits(qs)
        .depth(depth)
        .seed(seed)
        .min_weight(min_weight)
        .max_weight(max_weight)
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

    let mut g: Graph = c.to_graph();
    g.plug_inputs(&vec![BasisElem::Z0; qs]);

    let g_adj = &g.to_adjoint();
    g.plug(&observable);
    g.plug(g_adj);

    println!("g has T-count: {}", g.tcount());
    quizx::simplify::full_simp(&mut g);
    println!("g has reduced T-count: {}", g.tcount());

    let mut d = Decomposer::new(&g);
    d.with_full_simp();
    let d = d.decomp_parallel(3);
    let s1 = d.scalar.complex_value();

    println!("{:?}", s1);

    let a = ApproxDecomposer::new(SimpFunc::FullSimp, true);
    let s2 = a.run(&g, 0.05, &DumbTDecomposer);

    println!("{:?}", s2);

    Ok(())
}
