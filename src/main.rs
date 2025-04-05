mod data;
mod automl;
mod models;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    target: String,
}

fn main() {
    let args = Args::parse();

    match data::load_csv(&args.file, &args.target) {
        Ok((x, y)) => {
            println!("Loaded data! X shape = {:?}, y shape = {:?}", x.dim(), y.dim());
            automl::run(x, y);
        }
        Err(e) => {
            eprintln!("Failed to load CSV: {}", e);
        }
    }
}
