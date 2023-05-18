use crate::distance::euclidean;
use ndarray::{Array1, ArrayView1};
use std::vec::Vec;

pub struct VectorDatabase {
  vectors: Vec<Array1<f64>>,
}

impl Default for VectorDatabase {
  fn default() -> Self {
    Self::new()
  }
}

impl VectorDatabase {
  /// Creates a new `VectorDatabase`.
  ///
  /// # Examples
  ///
  /// ```
  /// use rustyvectors::vector_database::VectorDatabase;
  ///
  /// let db = VectorDatabase::new();
  /// ```
  pub fn new() -> Self {
    Self {
      vectors: Vec::new(),
    }
  }

  /// Adds a vector to the database.
  ///
  /// # Examples
  ///
  /// ```
  /// use rustyvectors::vector_database::VectorDatabase;
  ///
  /// let mut db = VectorDatabase::new();
  /// db.add(&[1.0, 2.0, 3.0]);
  /// ```
  pub fn add(&mut self, vector: &[f64]) {
    let new_vector =
      Array1::from_shape_vec(vector.len(), vector.to_vec()).unwrap();
    self.vectors.push(new_vector);
  }

  // Removes a vector from the database by its index.
  ///
  /// # Examples
  ///
  /// ```
  /// use rustyvectors::vector_database::VectorDatabase;
  ///
  /// let mut db = VectorDatabase::new();
  /// db.add(&[1.0, 2.0, 3.0]);
  /// let removed_vector = db.remove(0);
  /// ```
  pub fn remove(&mut self, index: usize) -> Option<Array1<f64>> {
    if index < self.vectors.len() {
      Some(self.vectors.remove(index))
    } else {
      None
    }
  }

  /// Finds the index of the nearest vector to the given query vector.
  ///
  /// # Examples
  ///
  /// ```
  /// use rustyvectors::vector_database::VectorDatabase;
  ///
  /// let mut db = VectorDatabase::new();
  /// db.add(&[1.0, 2.0, 3.0]);
  /// db.add(&[4.0, 5.0, 6.0]);
  /// let index = db.nearest(&[1.0, 2.0, 3.0]);
  /// ```
  pub fn nearest(&self, query: &[f64]) -> Option<usize> {
    if self.vectors.is_empty() {
      return None;
    }

    let query_array = ArrayView1::from(query);
    let (index, _) = self
      .vectors
      .iter()
      .enumerate()
      .map(|(i, v)| (i, euclidean(&v.view(), &query_array)))
      .min_by(|&(_, dist1), &(_, dist2)| {
        dist1
          .partial_cmp(&dist2)
          .unwrap_or(std::cmp::Ordering::Equal)
      })?;

    Some(index)
  }

  /// Retrieves a vector from the database by its index.
  ///
  /// # Examples
  ///
  /// ```
  /// use rustyvectors::vector_database::VectorDatabase;
  ///
  /// let mut db = VectorDatabase::new();
  /// db.add(&[1.0, 2.0, 3.0]);
  /// let vector = db.get(0);
  /// ```
  pub fn get(&self, index: usize) -> Option<&Array1<f64>> {
    if index < self.vectors.len() {
      Some(self.vectors.get(index).unwrap())
    } else {
      None
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_default() {
    let db = VectorDatabase::default();
    assert_eq!(db.get(0), None);
  }

  #[test]
  fn test_add() {
    let mut db = VectorDatabase::new();
    db.add(&[1.0, 2.0, 3.0]);
    assert_eq!(db.get(0), Some(&ndarray::arr1(&[1.0, 2.0, 3.0])));
  }

  #[test]
  fn test_get() {
    let mut db = VectorDatabase::new();
    db.add(&[1.0, 2.0, 3.0]);
    assert_eq!(db.get(0), Some(&ndarray::arr1(&[1.0, 2.0, 3.0])));
  }

  #[test]
  fn test_nearest() {
    let mut db = VectorDatabase::new();
    db.add(&[1.0, 2.0, 3.0]);
    db.add(&[4.0, 5.0, 6.0]);
    assert_eq!(db.nearest(&[1.0, 2.0, 3.0]), Some(0));

    assert_eq!(db.nearest(&[4.0, 5.0, 6.0]), Some(1));
  }

  #[test]
  fn test_remove() {
    let mut db = VectorDatabase::new();
    db.add(&[1.0, 2.0, 3.0]);
    db.add(&[4.0, 5.0, 6.0]);
    assert_eq!(db.remove(0), Some(ndarray::arr1(&[1.0, 2.0, 3.0])));
    assert_eq!(db.get(0), Some(&ndarray::arr1(&[4.0, 5.0, 6.0])));
  }

  #[test]
  fn test_digit_recognition() {
    let mut db = VectorDatabase::new();

    // adding reference "images" (here simplified as 1D arrays of length 64)
    // normally, these would be preprocessed image data.
    db.add(&[1.0; 64]); // imagine this is a '0'
    db.add(&[2.0; 64]); // imagine this is a '1'

    // Now we get a new "image" and want to recognize which digit it is
    let new_image = [1.05; 64]; // this should be recognized as '0'
    let nearest = db.nearest(&new_image);
    assert_eq!(nearest, Some(0));

    let new_image = [2.05; 64]; // this should be recognized as '1'
    let nearest = db.nearest(&new_image);
    assert_eq!(nearest, Some(1));
  }
}
