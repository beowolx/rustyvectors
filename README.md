# RustyVectors: A Minimalist Vector Database in Rust

RustyVectors is a simple, yet powerful vector database built in Rust. Vector databases are a type of database that allows efficient storage and retrieval of vector data. They are particularly useful in machine learning and data analysis workflows where high-dimensional vectors are common. 

## Overview

This Rust-based implementation provides the foundational operations of a vector database:
- Addition of vectors
- Removal of vectors
- Query for the nearest vector
- Access to vectors by index

The `nearest` function uses the Euclidean distance metric to measure similarity and find the closest vector to a given query vector. This is a simple and effective method for many use cases, but it has limitations. It assumes that all dimensions are equally important and may not perform well in very high-dimensional spaces due to the "curse of dimensionality". 

## Use Cases

RustyVectors can be applied in a variety of scenarios where efficient vector operations are required. Some examples include:

- **Image Recognition**: Vector databases can be used to store preprocessed image data. When a new image is processed, it can be compared to the existing vectors in the database to identify similar images or classify the image based on the closest match. 

- **Natural Language Processing**: Word embeddings, which represent words as high-dimensional vectors, can be stored and searched in this database to find similar words or analyze semantic relationships between words.

- **Recommendation Systems**: Product or user embeddings can be stored in this database. Given a user or a product, the database can quickly find other users or products that are most similar.

## Getting Started

To use RustyVectors, simply clone the repository and import the `VectorDatabase` struct into your project. You can then use the provided methods to manipulate and query your vector database.

```rust
let mut db = VectorDatabase::new();
db.add(&[1.0, 2.0, 3.0]);
println!("{:?}", db.nearest(&[1.0, 2.0, 3.0]));
```

## Future Improvements

While RustyVectors currently uses Euclidean distance for nearest neighbor search, future versions may support alternative distance metrics or implement approximate nearest neighbor search algorithms for improved performance in high-dimensional spaces. Contributions to this project are always welcome!

## Conclusion

RustyVectors is a minimalist, yet efficient tool for managing vector data in Rust. Whether you're working with machine learning models, building a recommendation system, or dealing with any other type of vector data, RustyVectors is a reliable and easy-to-use solution.
