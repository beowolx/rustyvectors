# RustyVectors: A Minimalist Vector Database in Rust

RustyVectors is a simple, yet powerful vector database built in Rust. Vector databases are a type of database that allows efficient storage and retrieval of vector data. They are particularly useful in machine learning and data analysis workflows where high-dimensional vectors are common. 

## Overview

This Rust-based implementation provides the foundational operations of a vector database:
- Addition of vectors
- Removal of vectors
- Query for the nearest vector
- Access to vectors by index

The `nearest` function uses the Euclidean distance metric to measure similarity and find the closest vector to a given query vector. This is a simple and effective method for many use cases, but it has limitations. It assumes that all dimensions are equally important and may not perform well in very high-dimensional spaces due to the "curse of dimensionality". 

## Getting Started

To use RustyVectors, simply clone the repository and import the `VectorDatabase` struct into your project. You can then use the provided methods to manipulate and query your vector database.

```rust
let mut db = VectorDatabase::new();
db.add(&[1.0, 2.0, 3.0]);
println!("{:?}", db.nearest(&[1.0, 2.0, 3.0]));
```

## Future Improvements

While RustyVectors currently uses Euclidean distance for nearest neighbor search, future versions may support alternative distance metrics or implement approximate nearest neighbor search algorithms for improved performance in high-dimensional spaces. Contributions to this project are always welcome!
