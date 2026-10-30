CREATE DATABASE college;
USE college;

CREATE TABLE students (
    id INT,
    name VARCHAR(50),
    branch VARCHAR(20)
);

INSERT INTO students VALUES
(1, 'Aanchal', 'ECE');

SELECT * FROM students;
  