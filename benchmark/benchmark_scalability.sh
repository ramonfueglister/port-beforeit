#!/bin/bash
# Speed-up with the number of threads, BeforeIT.jl
# benchmark/benchmark_scalability.sh: the example benchmark_scalability
# measures the model of Austria with 1 to 6 threads (at most the number of
# logical CPUs, nproc) and plots the speed-up over one thread. The files are
# in target/example-output/benchmark_scalability. Linux only (taskset).

set -e

cd "$(dirname "$0")/.."
cargo build --release --features plots --example benchmark_scalability
BIN=target/release/examples/benchmark_scalability

OUT_DIR=target/example-output/benchmark_scalability
mkdir -p "$OUT_DIR"
OUTPUT_CSV="$OUT_DIR/benchmark_results.csv"

if [ -f "$OUTPUT_CSV" ]; then
    echo "Removing existing $OUTPUT_CSV"
    rm "$OUTPUT_CSV"
fi

echo "Threads,BenchmarkType,MedianTime_ns" > "$OUTPUT_CSV"

# Julia runs 1 to 6 threads with julia --threads=$i; the port starts as
# many threads at once as the logical CPUs it may run on, which taskset
# limits to i.
CPUS=$(nproc)
MAX_THREADS=$(( CPUS < 6 ? CPUS : 6 ))

for i in $(seq 1 "$MAX_THREADS"); do
    echo "Running benchmarks with $i threads..."
    taskset -c "0-$((i - 1))" "$BIN" measure "$OUTPUT_CSV"
done

echo "Benchmarks completed and results saved to $OUTPUT_CSV"

"$BIN" plot "$OUTPUT_CSV"
