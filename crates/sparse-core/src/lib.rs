//! # SpTeCE Core Abstraction Layer
//!
//! This crate defines the fundamental traits and types for the Sparse Tensor Computation Engine (SpTeCE). It provides the abstract interface that all tensor implementations must satisfy.

use std::fmt::Debug;
use num_traits::{Num, Zero};
use thiserror::Error;

// ----------------------------------------------------------------------------
// Error Types
// ----------------------------------------------------------------------------

/// The primary error type for all tensor operations in SpTeCE.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Dimensions of operands are incompatible for the requested operation.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    /// An index is out of bounds for the tensor's shape.
    #[error("Index {index:?} out of bounds for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    
    /// The requested operation is not supported for the given tensor type or format.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    /// The tensor format cannot be converted to the requested format.
    #[error("Format conversion failed: {0}")]
    FormatConversion(String),
    
    /// Generic error for tensor operations.
    #[error("Tensor operation failed: {0}")]
    OperationFailed(String),
}

/// Result type alias for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;

// ----------------------------------------------------------------------------
// Data Type Enumeration
// ----------------------------------------------------------------------------

/// Supported numerical data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// Boolean values
    Bool,
}

impl DataType {
    /// Returns the size in bytes of this data type.
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::UInt8 => 1,
            DataType::Bool => 1,
        }
    }
    
    /// Returns true if the data type is floating point.
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }
    
    /// Returns true if the data type is integer.
    pub fn is_integer(&self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64 | DataType::UInt8)
    }
}

// ----------------------------------------------------------------------------
// Core Tensor Trait
// ----------------------------------------------------------------------------

/// The fundamental trait for all tensors in SpTeCE.
///
/// This trait defines the minimal interface that all tensor implementations
/// must provide, regardless of their storage format or sparsity pattern.
pub trait Tensor: Debug + Send + Sync {
    /// Returns the shape of the tensor as a slice.
    ///
    /// The shape is a vector where each element represents the size of
    /// the tensor along that dimension.
    fn shape(&self) -> &[usize];
    
    /// Returns the number of dimensions (rank) of the tensor.
    fn ndim(&self) -> usize {
        self.shape().len()
    }
    
    /// Returns the total number of elements in the tensor.
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    
    /// Returns the data type of the tensor elements.
    fn dtype(&self) -> DataType;
    
    /// Returns the value at the given indices, or an error if indices are invalid.
    ///
    /// # Arguments
    /// * `indices` - Slice of indices, one per dimension
    ///
    /// # Returns
    /// The value at the specified position, or an error if indices are out of bounds.
    fn get(&self, indices: &[usize]) -> TensorResult<f64>;
    
    /// Returns true if the tensor is sparse (has specialized storage for zeros).
    fn is_sparse(&self) -> bool;
    
    /// Converts the tensor to a dense representation.
    ///
    /// For sparse tensors, this will allocate memory for all elements,
    /// including zeros.
    fn to_dense(&self) -> TensorResult<Box<dyn Tensor>>;
}

// ----------------------------------------------------------------------------
// Sparse Tensor Trait
// ----------------------------------------------------------------------------

/// Extension trait for sparse tensors with specialized sparse operations.
pub trait SparseTensor: Tensor {
    /// Returns the number of non-zero elements in the tensor.
    fn nnz(&self) -> usize;
    
    /// Returns the sparsity ratio (0.0 = dense, 1.0 = all zeros).
    fn sparsity(&self) -> f64 {
        if self.numel() == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / self.numel() as f64)
        }
    }
    
    /// Returns an iterator over non-zero elements and their indices.
    fn nonzero_iter(&self) -> Box<dyn Iterator<Item = (Vec<usize>, f64)> + '_>;
    
    /// Returns the storage format of the sparse tensor.
    fn format(&self) -> SparseFormat;
}

// ----------------------------------------------------------------------------
// Sparse Format Enumeration
// ----------------------------------------------------------------------------

/// Supported sparse storage formats in SpTeCE.
///
/// Each format has different performance characteristics for various
/// operations and sparsity patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SparseFormat {
    /// Coordinate Format (COO)
    ///
    /// Stores tuples of (indices, value). Best for incremental construction
    /// and format conversion. Memory: O(nnz * ndim).
    Coordinate,
    
    /// Compressed Sparse Row (CSR)
    ///
    /// Compresses row indices for 2D matrices. Best for row-wise operations
    /// like SpMV. Memory: O(nnz + rows).
    CompressedSparseRow,
    
    /// Compressed Sparse Column (CSC)
    ///
    /// Compresses column indices for 2D matrices. Best for column-wise operations.
    /// Memory: O(nnz + cols).
    CompressedSparseColumn,
    
    /// Block Sparse Row (BSR)
    ///
    /// Like CSR but for dense sub-blocks. Best for structured sparsity.
    /// Memory: O(nnz + rows/block_size).
    BlockSparseRow,
    
    /// Diagonal Storage (DIA)
    ///
    /// Stores diagonals compactly. Best for banded matrices.
    /// Memory: O(min(rows, cols) * num_diags).
    Diagonal,
}

impl SparseFormat {
    /// Returns true if the format is efficient for the given operation.
    pub fn efficient_for(&self, operation: &SparseOperation) -> bool {
        match (self, operation) {
            (SparseFormat::CompressedSparseRow, SparseOperation::SpMV) => true,
            (SparseFormat::CompressedSparseColumn, SparseOperation::SpMV) => true,
            (SparseFormat::Coordinate, SparseOperation::Construction) => true,
            (SparseFormat::BlockSparseRow, SparseOperation::SpMM) => true,
            (SparseFormat::Diagonal, SparseOperation::SpMV) => true,
            _ => false,
        }
    }
    
    /// Returns the recommended format for a tensor with the given shape and sparsity pattern.
    pub fn recommend_for(shape: &[usize], sparsity_pattern: Option<&str>) -> Self {
        match shape.len() {
            0 => SparseFormat::Coordinate, // Scalar
            1 => SparseFormat::Coordinate, // Vector
            2 => {
                // For matrices, choose based on expected operations
                if let Some(pattern) = sparsity_pattern {
                    match pattern {
                        "banded" | "diagonal" => SparseFormat::Diagonal,
                        "block" => SparseFormat::BlockSparseRow,
                        _ => SparseFormat::CompressedSparseRow,
                    }
                } else {
                    SparseFormat::CompressedSparseRow
                }
            }
            _ => SparseFormat::Coordinate, // Higher-order tensors
        }
    }
}

// ----------------------------------------------------------------------------
// Sparse Operation Types
// ----------------------------------------------------------------------------

/// Common sparse tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseOperation {
    /// Sparse matrix-vector multiplication
    SpMV,
    /// Sparse matrix-matrix multiplication
    SpMM,
    /// Element-wise operations
    ElementWise,
    /// Tensor contraction
    Contraction,
    /// Format conversion
    Conversion,
    /// Tensor construction
    Construction,
    /// Slicing/indexing
    Slicing,
}

// ----------------------------------------------------------------------------
// Tensor Creation and Validation Utilities
// ----------------------------------------------------------------------------

/// Validates that indices are within the tensor's shape bounds.
pub fn validate_indices(indices: &[usize], shape: &[usize]) -> TensorResult<()> {
    if indices.len() != shape.len() {
        return Err(TensorError::DimensionMismatch(format!(
            "Expected {} indices for {}-dimensional tensor, got {}",
            shape.len(),
            shape.len(),
            indices.len()
        )));
    }
    
    for (dim, (&idx, &dim_size)) in indices.iter().zip(shape.iter()).enumerate() {
        if idx >= dim_size {
            return Err(TensorError::IndexOutOfBounds {
                index: indices.to_vec(),
                shape: shape.to_vec(),
            });
        }
    }
    
    Ok(())
}

/// Calculates the linear index from multi-dimensional indices and strides.
pub fn linear_index(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

/// Calculates strides for a given shape in row-major order.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ----------------------------------------------------------------------------
// Generic Tensor Implementation (for testing and prototyping)
// ----------------------------------------------------------------------------

/// A generic dense tensor implementation for testing and prototyping.
#[derive(Debug, Clone)]
pub struct DenseTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
    strides: Vec<usize>,
}

impl DenseTensor {
    /// Creates a new dense tensor from shape and data.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> TensorResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(TensorError::DimensionMismatch(format!(
                "Expected {} elements for shape {:?}, got {}",
                expected_len, shape, data.len()
            )));
        }
        
        let strides = compute_strides(&shape);
        
        Ok(Self {
            shape,
            data,
            strides,
        })
    }
    
    /// Creates a zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; numel],
            strides: compute_strides(&shape),
        }
    }
}

impl Tensor for DenseTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn dtype(&self) -> DataType {
        DataType::Float64
    }
    
    fn get(&self, indices: &[usize]) -> TensorResult<f64> {
        validate_indices(indices, &self.shape)?;
        let idx = linear_index(indices, &self.strides);
        Ok(self.data[idx])
    }
    
    fn is_sparse(&self) -> bool {
        false
    }
    
    fn to_dense(&self) -> TensorResult<Box<dyn Tensor>> {
        Ok(Box::new(self.clone()))
    }
}

// ----------------------------------------------------------------------------
// Unit Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::UInt8.size(), 1);
    }
    
    #[test]
    fn test_data_type_classification() {
        assert!(DataType::Float32.is_float());
        assert!(DataType::Float64.is_float());
        assert!(!DataType::Float32.is_integer());
        assert!(DataType::Int32.is_integer());
    }
    
    #[test]
    fn test_validate_indices() {
        let shape = vec![3, 4, 5];
        
        // Valid indices
        assert!(validate_indices(&[0, 0, 0], &shape).is_ok());
        assert!(validate_indices(&[2, 3, 4], &shape).is_ok());
        
        // Invalid: wrong number of indices
        assert!(validate_indices(&[0, 0], &shape).is_err());
        assert!(validate_indices(&[0, 0, 0, 0], &shape).is_err());
        
        // Invalid: out of bounds
        assert!(validate_indices(&[3, 0, 0], &shape).is_err());
        assert!(validate_indices(&[0, 4, 0], &shape).is_err());
        assert!(validate_indices(&[0, 0, 5], &shape).is_err());
    }
    
    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 2]), vec![2, 1]);
    }
    
    #[test]
    fn test_linear_index() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(linear_index(&[0, 0, 0], &strides), 0);
        assert_eq!(linear_index(&[1, 0, 0], &strides), 12);
        assert_eq!(linear_index(&[0, 1, 0], &strides), 4);
        assert_eq!(linear_index(&[0, 0, 1], &strides), 1);
        assert_eq!(linear_index(&[1, 2, 3], &strides), 12 + 8 + 3);
    }
    
    #[test]
    fn test_dense_tensor() {
        let tensor = DenseTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.dtype(), DataType::Float64);
        assert!(!tensor.is_sparse());
        
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
        
        // Test error cases
        assert!(tensor.get(&[2, 0]).is_err());
        assert!(tensor.get(&[0, 3]).is_err());
        assert!(tensor.get(&[0]).is_err());
    }
    
    #[test]
    fn test_sparse_format_recommendation() {
        // 2D tensors (matrices)
        assert_eq!(
            SparseFormat::recommend_for(&[100, 100], None),
            SparseFormat::CompressedSparseRow
        );
        
        assert_eq!(
            SparseFormat::recommend_for(&[100, 100], Some("diagonal")),
            SparseFormat::Diagonal
        );
        
        assert_eq!(
            SparseFormat::recommend_for(&[100, 100], Some("block")),
            SparseFormat::BlockSparseRow
        );
        
        // Higher-order tensors
        assert_eq!(
            SparseFormat::recommend_for(&[10, 10, 10], None),
            SparseFormat::Coordinate
        );
        
        // Vectors and scalars
        assert_eq!(
            SparseFormat::recommend_for(&[100], None),
            SparseFormat::Coordinate
        );
        
        assert_eq!(
            SparseFormat::recommend_for(&[], None),
            SparseFormat::Coordinate
        );
    }
    
    #[test]
    fn test_sparse_format_efficiency() {
        assert!(SparseFormat::CompressedSparseRow.efficient_for(&SparseOperation::SpMV));
        assert!(SparseFormat::Coordinate.efficient_for(&SparseOperation::Construction));
        assert!(SparseFormat::BlockSparseRow.efficient_for(&SparseOperation::SpMM));
        
        // Not efficient for these operations
        assert!(!SparseFormat::Diagonal.efficient_for(&SparseOperation::SpMM));
    }
}