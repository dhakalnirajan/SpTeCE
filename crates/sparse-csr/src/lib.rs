//! # CSR (Compressed Sparse Row) Format for SpTeCE
//!
//! This crate provides the CSR format implementation for 2D sparse matrices.
//! CSR is optimal for row-wise operations like SpMV (Sparse Matrix-Vector Multiply).

use std::ops::{Add, Mul, Sub};
use std::fmt;

use num_traits::{Num, Zero, PrimInt, ToPrimitive};
use sparse_core::*;

// Re-export your original CSR implementation
pub use csr_matrix::CsrMatrix;

mod csr_matrix;

// ----------------------------------------------------------------------------
// CSR Tensor Adapter
// ----------------------------------------------------------------------------

/// Adapter that wraps a CSR matrix to implement the `SparseTensor` trait.
#[derive(Debug, Clone)]
pub struct CsrTensor<T, I = usize>
where
    T: Num + Copy + Default + PartialEq + fmt::Debug + Send + Sync,
    I: PrimInt + Zero + Copy + fmt::Debug + ToPrimitive + Send + Sync,
{
    matrix: CsrMatrix<T, I>,
}

impl<T, I> CsrTensor<T, I>
where
    T: Num + Copy + Default + PartialEq + fmt::Debug + Send + Sync + Into<f64>,
    I: PrimInt + Zero + Copy + fmt::Debug + ToPrimitive + Send + Sync,
{
    /// Creates a new CSR tensor from a CSR matrix.
    pub fn new(matrix: CsrMatrix<T, I>) -> Self {
        Self { matrix }
    }
    
    /// Returns a reference to the underlying CSR matrix.
    pub fn inner(&self) -> &CsrMatrix<T, I> {
        &self.matrix
    }
    
    /// Consumes the wrapper and returns the underlying CSR matrix.
    pub fn into_inner(self) -> CsrMatrix<T, I> {
        self.matrix
    }
}

impl<T, I> Tensor for CsrTensor<T, I>
where
    T: Num + Copy + Default + PartialEq + fmt::Debug + Send + Sync + Into<f64>,
    I: PrimInt + Zero + Copy + fmt::Debug + ToPrimitive + Send + Sync,
{
    fn shape(&self) -> &[usize] {
        // CSR matrices are always 2D
        &[self.matrix.rows(), self.matrix.cols()]
    }
    
    fn dtype(&self) -> DataType {
        // Map Rust types to DataType enum
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            DataType::Float32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            DataType::Float64
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            DataType::Int32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            DataType::Int64
        } else {
            // Default to Float64 for unknown types
            DataType::Float64
        }
    }
    
    fn get(&self, indices: &[usize]) -> TensorResult<f64> {
        validate_indices(indices, self.shape())?;
        
        if indices.len() != 2 {
            return Err(TensorError::DimensionMismatch(format!(
                "CSR tensors require 2 indices (row, col), got {}",
                indices.len()
            )));
        }
        
        let row = indices[0];
        let col = indices[1];
        
        // Use the binary search implementation from your code
        let value = self.matrix.get(row, col);
        Ok(value.into())
    }
    
    fn is_sparse(&self) -> bool {
        true
    }
    
    fn to_dense(&self) -> TensorResult<Box<dyn Tensor>> {
        let (rows, cols) = self.matrix.shape();
        let mut data = vec![0.0; rows * cols];
        
        // Fill dense representation
        for row in 0..rows {
            let start = self.matrix.row_pointers[row].to_usize().unwrap_or(0);
            let end = self.matrix.row_pointers[row + 1].to_usize().unwrap_or(self.matrix.values.len());
            
            for idx in start..end {
                let col = self.matrix.column_indices[idx].to_usize().unwrap_or(0);
                let value: f64 = self.matrix.values[idx].into();
                data[row * cols + col] = value;
            }
        }
        
        Ok(Box::new(DenseTensor::new(vec![rows, cols], data)?))
    }
}

impl<T, I> SparseTensor for CsrTensor<T, I>
where
    T: Num + Copy + Default + PartialEq + fmt::Debug + Send + Sync + Into<f64>,
    I: PrimInt + Zero + Copy + fmt::Debug + ToPrimitive + Send + Sync,
{
    fn nnz(&self) -> usize {
        self.matrix.nnz()
    }
    
    fn nonzero_iter(&self) -> Box<dyn Iterator<Item = (Vec<usize>, f64)> + '_> {
        let rows = self.matrix.rows();
        let cols = self.matrix.cols();
        
        let iter = (0..rows).flat_map(move |row| {
            let start = self.matrix.row_pointers[row].to_usize().unwrap_or(0);
            let end = self.matrix.row_pointers[row + 1].to_usize().unwrap_or(self.matrix.values.len());
            
            (start..end).map(move |idx| {
                let col = self.matrix.column_indices[idx].to_usize().unwrap_or(0);
                let value: f64 = self.matrix.values[idx].into();
                (vec![row, col], value)
            })
        });
        
        Box::new(iter)
    }
    
    fn format(&self) -> SparseFormat {
        SparseFormat::CompressedSparseRow
    }
}

// ----------------------------------------------------------------------------
// CSR Matrix Module (Your Original Code)
// ----------------------------------------------------------------------------

mod csr_matrix {
    // Your original CSR implementation goes here
    // I'll include a minimal version for completeness
    
    use std::ops::{Add, Mul, Sub};
    use std::fmt;
    
    use num_traits::{Num, Zero, PrimInt, ToPrimitive};
    use serde::{Serialize, Deserialize};
    
    /// Represents errors that can occur during CsrMatrix operations.
    #[derive(Debug, Clone, PartialEq)]
    pub enum MatrixError {
        InvalidInputDimensions,
        IndexOutOfBounds,
        DimensionMismatch,
        IndexTypeOverflow,
    }
    
    impl fmt::Display for MatrixError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                MatrixError::InvalidInputDimensions => write!(f, "Error: Input dimensions are invalid (e.g., jagged matrix)."),
                MatrixError::IndexOutOfBounds => write!(f, "Error: Index is out of matrix bounds or element not stored."),
                MatrixError::DimensionMismatch => write!(f, "Error: Matrix dimensions do not match for this operation."),
                MatrixError::IndexTypeOverflow => write!(f, "Error: Matrix size or number of non-zero elements exceeds the capacity of the index type."),
            }
        }
    }
    
    /// A generic sparse matrix represented in Compressed Sparse Row (CSR) format.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CsrMatrix<T, I = usize> {
        rows: usize,
        cols: usize,
        pub values: Vec<T>,
        pub column_indices: Vec<I>,
        pub row_pointers: Vec<I>,
    }
    
    impl<T, I> CsrMatrix<T, I>
    where
        T: Num + Copy + Default + PartialEq + fmt::Debug + Send + Sync,
        I: PrimInt + Zero + Copy + fmt::Debug + ToPrimitive + Send + Sync,
    {
        // Private helper function to safely convert index type I to usize
        fn safe_to_usize(&self, val: I) -> Result<usize, MatrixError> {
            val.to_usize().ok_or(MatrixError::IndexTypeOverflow)
        }
        
        /// Creates a new, empty CsrMatrix with specified dimensions.
        pub fn new(rows: usize, cols: usize) -> Self {
            let row_pointers = vec![I::zero(); rows + 1];
            
            Self {
                rows,
                cols,
                values: Vec::new(),
                column_indices: Vec::new(),
                row_pointers,
            }
        }
        
        /// Creates a CsrMatrix from a standard dense representation (`&[Vec<T>]`).
        pub fn from_dense(dense: &[Vec<T>]) -> Result<Self, MatrixError> {
            let rows = dense.len();
            let cols = dense.first().map_or(0, |row| row.len());
            
            if dense.iter().any(|row| row.len() != cols) {
                return Err(MatrixError::InvalidInputDimensions);
            }
            
            let mut values = Vec::new();
            let mut column_indices = Vec::new();
            let mut row_pointers = vec![I::zero(); rows + 1];
            
            let mut nnz: usize = 0;
            
            for (i, row) in dense.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    if val != T::zero() { 
                        values.push(val);
                        let j_index = I::from(j).ok_or(MatrixError::IndexTypeOverflow)?;
                        column_indices.push(j_index);
                        nnz += 1;
                    }
                }
                let nnz_index = I::from(nnz).ok_or(MatrixError::IndexTypeOverflow)?;
                row_pointers[i + 1] = nnz_index;
            }
            
            Ok(Self { rows, cols, values, column_indices, row_pointers })
        }
        
        /// Safely retrieves the value at position (i, j). Returns T::zero() if not stored or out of bounds.
        /// Implements **Binary Search** for $O(\log n)$ access time.
        pub fn get(&self, i: usize, j: usize) -> T {
            if i >= self.rows || j >= self.cols {
                return T::zero();
            }
            
            let start = match self.safe_to_usize(self.row_pointers[i]) {
                Ok(idx) => idx,
                Err(_) => return T::zero(), 
            };
            let end = match self.safe_to_usize(self.row_pointers[i + 1]) {
                Ok(idx) => idx,
                Err(_) => return T::zero(),
            };
            
            let j_index = I::from(j);
            if j_index.is_none() { return T::zero(); }
            let j_index = j_index.unwrap();
            
            // Binary search on the column indices slice
            let row_cols = &self.column_indices[start..end];
            if let Ok(local_idx) = row_cols.binary_search(&j_index) {
                return self.values[start + local_idx];
            }
            
            T::zero()
        }
        
        pub fn nnz(&self) -> usize { self.values.len() }
        pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
        pub fn rows(&self) -> usize { self.rows }
        pub fn cols(&self) -> usize { self.cols }
        
        /// Computes the transpose of the matrix, $A^T$.
        pub fn transpose(&self) -> Result<CsrMatrix<T, I>, MatrixError> {
            let (rows, cols) = (self.rows, self.cols);
            let nnz = self.nnz();
            
            let mut col_counts = vec![0; cols];
            let mut row_pointers_t = vec![I::zero(); cols + 1];
            let mut column_indices_t = vec![I::zero(); nnz];
            let mut values_t = vec![T::zero(); nnz];
            
            // Count non-zero elements per column
            for &col_index in self.column_indices.iter() {
                let j = self.safe_to_usize(col_index)?;
                col_counts[j] += 1;
            }
            
            // Determine row pointers for A^T
            let mut current_offset = I::zero();
            for j in 0..cols {
                row_pointers_t[j + 1] = current_offset + I::from(col_counts[j]).ok_or(MatrixError::IndexTypeOverflow)?;
                current_offset = row_pointers_t[j + 1];
            }
            
            let mut current_row_offsets = row_pointers_t.clone();
            
            // Fill the values and column indices for A^T
            for i in 0..rows {
                let start = self.safe_to_usize(self.row_pointers[i])?;
                let end = self.safe_to_usize(self.row_pointers[i + 1])?;
                
                for idx in start..end {
                    let j = self.safe_to_usize(self.column_indices[idx])?;
                    let insert_idx_i = self.safe_to_usize(current_row_offsets[j])?;
                    
                    values_t[insert_idx_i] = self.values[idx];
                    column_indices_t[insert_idx_i] = I::from(i).ok_or(MatrixError::IndexTypeOverflow)?;
                    
                    current_row_offsets[j] = current_row_offsets[j] + I::one();
                }
            }
            
            Ok(CsrMatrix {
                rows: cols,
                cols: rows,
                values: values_t,
                column_indices: column_indices_t,
                row_pointers: row_pointers_t,
            })
        }
    }
    
    // Matrix multiplication implementation
    impl<T, I> Mul<Vec<T>> for &CsrMatrix<T, I>
    where
        T: Num + Copy + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        I: PrimInt + Zero + Copy + ToPrimitive + Send + Sync,
    {
        type Output = Vec<T>;
        
        fn mul(self, rhs: Vec<T>) -> Self::Output {
            if self.cols != rhs.len() {
                panic!("{}", MatrixError::DimensionMismatch);
            }
            
            let mut y = vec![T::zero(); self.rows];
            
            for i in 0..self.rows {
                let start = self.row_pointers[i].to_usize().unwrap_or(0);
                let end = self.row_pointers[i + 1].to_usize().unwrap_or(self.values.len());
                
                let mut row_sum = T::zero();
                
                for idx in start..end {
                    let col_j = self.column_indices[idx].to_usize().unwrap_or(0);
                    row_sum = row_sum + self.values[idx] * rhs[col_j];
                }
                y[i] = row_sum;
            }
            
            y
        }
    }
}