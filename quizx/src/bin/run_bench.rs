use std::path::PathBuf;

use num::Complex;
use quizx::decompose::approximate::{ApproxDecomposer, DumbTDecomposer};
use quizx::decompose::{Decomposer, SimpFunc};
use quizx::json::read_graph;
use quizx::vec_graph::Graph;

use clap::Parser;

#[derive(Parser)]
#[command(version, about = "Running simulation benchmarks", long_about = None)]
struct Cli {
    #[arg(value_name = "FILE")]
    file: PathBuf,

    #[arg(short, long)]
    epsilon: Option<f64>,

    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    verbose: bool,

    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    simp: bool,

    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    parallel: bool,
}

fn main() {
    let cli = Cli::parse();

    let g: Graph = read_graph(&cli.file).unwrap();
    let simp = if cli.simp {
        SimpFunc::FullSimp
    } else {
        SimpFunc::NoSimp
    };

    let scalar: Complex<f64>;
    let terms: usize;
    if let Some(epsilon) = cli.epsilon {
        let decomposer = ApproxDecomposer::new(simp, cli.parallel);
        scalar = decomposer.run(&g, epsilon, &DumbTDecomposer);
        terms = 0;
    } else {
        let mut decomposer = Decomposer::new(&g);
        decomposer.with_simp(simp).use_cats(true).decomp_all();
        scalar = decomposer.scalar.complex_value();
        terms = decomposer.nterms;
    }

    if cli.verbose {
        println!("{terms}, {scalar}");
    } else {
        println!("{scalar}");
    }
}
