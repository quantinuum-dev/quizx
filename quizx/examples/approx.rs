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
    let qs = 20;
    let c = Circuit::random()
        .qubits(qs)
        .depth(500)
        .seed(1337)
        .p_t(0.05)
        .with_cliffords()
        .build();
    let mut g: Graph = c.to_graph();
    g.plug_inputs(&vec![BasisElem::Z0; qs]);
    g.plug_outputs(&vec![BasisElem::Z1; qs]);

    println!("g has T-count: {}", g.tcount() / 2);
    quizx::simplify::full_simp(&mut g);
    println!("g has reduced T-count: {}", g.tcount());

    let mut d = Decomposer::new(&g);
    d.with_full_simp();
    let d = d.decomp_parallel(3);
    let s1 = d.scalar.complex_value();

    println!("{:?}", s1 * s1.conj());

    let a = ApproxDecomposer::new(SimpFunc::FullSimp);
    let iters = 2.0f64.powf(0.22 * g.tcount() as f64);
    println!("{iters} iters");
    let s2 = a.run(&g.clone(), iters as usize, &DumbTDecomposer);

    println!("{:?}", s2 * s2.conj());

    Ok(())
}
