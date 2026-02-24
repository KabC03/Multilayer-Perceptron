#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>


/**
 * NOTE: This is a CPU-based library
 * Its intended for use with small perceptrons
 * std::thread could be used for larger matricies
 */


namespace matrix {
    template <typename T>
    class Matrix {
        private:
        size_t rows, cols;

        //For transpose and mmul (operations can't be done in place)
        //This avoids constant reallocation
        std::vector<T> data;
        public:

        /**
         * @brief Boring constructor for matrix class
         * 
         * @param rows :: Rows of the matrix
         * @param cols :: Cols of the matrix
         */
        Matrix(void) {
            this->rows = 0;
            this->cols = 0;
        }

        /**
         * @brief Initialise a matrix and reserve memory for its buffers
         * 
         * @param rows :: Rows of the matrix
         * @param cols :: Cols of the matrix
         */
        Matrix(size_t rows, size_t cols) {
            this->rows = rows;
            this->cols = cols;
            data.resize(rows * cols);
        }

       /**
        * @brief Fill a matrix with a vector (pre-dimensioned)
        * 
        * @param vec :: Vector to copy from
        *
        * @return void :: None
        */
        void fill_vector(std::vector<T> &vec) {
            this->data = vec;
            return;
        } 


       /**
        * @brief Getter for a matrix's data
        *
        * @return const std::vector<float> & :: Const reference to internal vector
        */
        const std::vector<float> &get_vector(void) {
            return this->data;;
        } 


        /**
         * @brief Refer to a element in a matrix
         * 
         * @param row :: Row of element
         * @param col :: Col of element
         * 
         * @return const T& :: Constant reference to data
         */
        inline T& at(size_t row, size_t col) {
            return this->data[row * this->cols + col];
        }
        inline const T& at(size_t row, size_t col) const {
            return this->data[row * this->cols + col];
        }



        /**
         * @brief In place matrix multiplication (z = xw)
         * 
         * @param x :: X matrix
         * @param w :: W matrix
         * 
         * @return Matrix<T>& :: Result matrix (this)
         */
        Matrix<T> &multiply(Matrix<T> &x, Matrix<T> &w, bool transposeX, bool transposeW) {
            //Multiply

            size_t xr = transposeX ? x.cols : x.rows;
            size_t xc = transposeX ? x.rows : x.cols;

            size_t wr = transposeW ? w.cols : w.rows;
            size_t wc = transposeW ? w.rows : w.cols;

            for(size_t i = 0; i < xr; i++) {
                for(size_t j = 0; j < wc; j++) {

                    this->at(i, j) = 0;
                    for(size_t k = 0; k < xc; k++) {

                        T xVal = transposeX ? x.at(k, i) : x.at(i, k);
                        T wVal = transposeW ? w.at(j, k) : w.at(k, j);

                        this->at(i, j) += xVal * wVal;
                    }
                }
            }
            return *this;
        }


        /**
         * @brief Bias a matrix in place
         * 
         * @param x :: Bias to add
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &add(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] += rhs.data[i];
            }
            return *this;
        }

        /**
         * @brief Negative bias a matrix in place
         * 
         * @param rhs :: Negative bias to add (rhs)
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &subtract(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] -= rhs.data[i];
            }
            return *this;
        }

        /**
         * @brief Hadamard a matrix in place (negative)
         * 
         * @param rhs :: Hadamard rhs
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &hadamard(Matrix<T> &rhs) {
            for(size_t i = 0; i < rhs.rows * rhs.cols; i++) {
                this->data[i] *= rhs.data[i];
            }
            return *this;
        }


        /**
         * @brief Get the number of rows in a matrix
         * 
         * @return Number of rows in the matrix
         */
        size_t get_rows(void) {
            return this->rows;
        }


        /**
         * @brief Get the number of cols in a matrix
         * 
         * @return Number of cols in the matrix
         */
        size_t get_cols(void) {
            return this->cols;
        }

        /**
         * @brief Multiply a matrix by a scalar 
         * 
         * @param scalar :: Scalar to multiply by 
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &scalar_multiply(T scalar) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] *= scalar;
            }
            return *this;
        }


        /**
         * @brief Rowwise addition of two matricies
         * 
         * @param matrix :: Row to add
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &add_row(Matrix<T> &matrix) {
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    this->at(i, j) += matrix.at(0, j);
                }
            }
            return *this;
        }


        /**
         * @brief Append a matrix to a file, including dimensions
         * 
         * @param fileName :: File name
         * 
         * @return bool :: Indication of if matrix was successfully written (true = yes)
         */
        bool append_to_file(std::string fileName) {
            std::ofstream file(fileName, std::ios::app);
            if(!file) {
                return false;
            }
            file << this->rows << "\n" << this->cols << "\n";
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    file << this->at(i, j) << " ";
                }
            }
            file << "\n";
            return file.good(); //Make sure file is ok
        }


        /**
         * @brief Read a matrix from a file stream. Expects rows,\n,cols,\n,data
         * 
         * @param fstream :: File name
         * 
         * @return bool :: Indication of if matrix was successfully read (true = yes)
         */
        bool read_float_file_fstream(std::ifstream &file) {
            if(!file) {
                return false;
            }
            

            //Get dimensions
            std::string line;
            std::getline(file, line);
            size_t rows = std::stoull(line);
            std::getline(file, line);
            size_t cols = std::stoull(line);
            this->rows = rows;
            this->cols = cols;
            this->data.resize(rows * cols);

            //Get data
            for(size_t i = 0; i < rows; i++) {
                for(size_t j = 0; j < cols; j++) {
                    std::getline(file, line, ' ');
                    this->data[i * cols + j] = std::stof(line);
                }
            }
            //Skip \n
            std::getline(file, line);

            return true;
        }


        /**
         * @brief Sum accross all rows in a matrix
         * 
         * @param matrix :: Rows to sum accross
         * 
         * @return Matrix<T>& :: This matrix (result)
         */
        Matrix<T> &sum_rows(Matrix<T> &matrix) {
            for(size_t i = 0; i < this->rows; i++) {
                for(size_t j = 0; j < this->cols; j++) {
                    this->at(0, j) += matrix.at(i, j);
                }
            }
            return *this;
        }
        

        /**
         * @brief ReLU activation on a matrix
         * 
         * @return Matrix<T>& :: This (result matrix)
         */
        Matrix<T> &activate_relu(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = (this->data[i] > 0) * this->data[i];
            }
            return *this;
        }
        /**
         * @brief ReLU derivateive activation on a matrix
         * 
         * @return Matrix<T>& :: This (result matrix)
         */
        Matrix<T> &activate_derivative_relu(void) {
            for(size_t i = 0; i < this->rows * this->cols; i++) {
                this->data[i] = (this->data[i] > 0);
            }
            return *this;
        }

    };
}



#endif

