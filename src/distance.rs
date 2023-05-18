use ndarray::ArrayView1;

/// Computes the Euclidean distance between two 1-dimensional array views.
///
/// # Arguments
/// 
/// * `a` - The first array view.
/// * `b` - The second array view.
///
/// # Examples
///
/// ```
/// use rustyvectors::distance::euclidean;
/// use ndarray::{arr1, ArrayView1};
///
/// let array1 = arr1(&[1.0, 2.0, 3.0]);
/// let array2 = arr1(&[4.0, 5.0, 6.0]);
/// let dist = euclidean(&array1.view(), &array2.view());
/// ```
pub fn euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| (x - y).powi(2))
    .sum::<f64>()
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_zero_distance_same_vectors() {
        let vec1 = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let vec2 = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(euclidean(&vec1.view(), &vec2.view()), 0.0);
    }

    #[test]
    fn test_known_distance_diff_vectors() {
        let vec1 = arr1(&[1.0, 2.0, 3.0]);
        let vec2 = arr1(&[4.0, 5.0, 6.0]);

        assert_eq!(euclidean(&vec1.view(), &vec2.view()), (27.0 as f64).sqrt());
    }

    #[test]
    fn test_distance_invariant_to_scaling() {
        let vec1 = arr1(&[1.0, 2.0, 3.0]);
        let vec2 = arr1(&[2.0, 4.0, 6.0]);

        assert_eq!(euclidean(&vec1.view(), &vec2.view()), (14.0 as f64).sqrt());
    }
}
