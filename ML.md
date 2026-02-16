# What is Machine Learning (ML)?

## Definition

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn from data and make predictions or decisions without being explicitly programmed.

### In simple words:

> Instead of writing rules manually, we give data to the machine and it learns patterns automatically.

---

## Why Machine Learning?

### Traditional Programming

Input + Rules → Output

### Machine Learning

Input + Output → Machine learns Rules

### ML is useful when:

- Rules are complex
- Data is large
- Patterns are difficult to write manually

### Examples

- Email spam detection
- Face recognition
- Recommendation systems
- Chatbots

---

## Types of Machine Learning

### 1. Supervised Learning

The model is trained using labeled data (input + correct output).

#### Regression
Used when output is continuous (numbers).

Examples:
- House price prediction
- Salary prediction

#### Classification
Used when output is categorical (Yes/No, 0/1).

Examples:
- Spam detection
- Disease prediction

---

### 2. Unsupervised Learning

The model is trained using data without labels.

Examples:
- Customer segmentation
- Market grouping

---

### 3. Reinforcement Learning

The model learns using rewards and penalties.

Examples:
- Self-driving cars
- Game playing AI

---

## Important ML Terminologies

- **Dataset** – Collection of data used to train the model
- **Features (X)** – Input variables
- **Target (y)** – Output variable
- **Training Data** – Data used to train the model
- **Testing Data** – Data used to evaluate performance
- **Overfitting** – Model performs well on training data but poorly on new data
- **Underfitting** – Model fails to learn properly
- **Accuracy** – Percentage of correct predictions
- **Feature Scaling** – Normalizing data to similar scale

---

## Basic ML Workflow

1. Collect Data  
2. Clean Data  
3. Split Data  
4. Train Model  
5. Evaluate Model  
6. Improve Model  
7. Deploy Model  

---

## Conclusion

Machine Learning allows systems to:

- Learn from data
- Identify patterns
- Make predictions
- Improve over time

# 🔢 Basic Calculations Used in Machine Learning

Machine Learning internally uses mathematics to learn patterns.

---

## 1. Mean (Average)

Data no average value.

### Formula

Mean = (Sum of all values) / (Total values)

### Example

Marks: 50, 60, 70  

Mean = (50 + 60 + 70) / 3  
Mean = 60  

Used for:
- Understanding data
- Filling missing values
- Normalization

---

## 2. Variance

Measures how spread out the data is from the mean.

### Formula

Variance = Σ(x - mean)² / n

High variance → data widely spread  
Low variance → data close to mean  

---

## 3. Standard Deviation

Square root of variance.

Standard Deviation = √Variance

Used in:
- Feature scaling
- Data normalization

---

## 4. Linear Regression Basic Formula

y = mx + c

Where:
- y = prediction  
- x = input  
- m = slope  
- c = intercept  

### Example

m = 2  
c = 5  
x = 3  

y = 2(3) + 5  
y = 11  

---

## 5. Loss Function (Error Calculation)

Measures how wrong the model is.

### Mean Squared Error (MSE)

MSE = Σ(Actual - Predicted)² / n

Example:

Actual = 10  
Predicted = 8  

Error = (10 - 8)²  
Error = 4  

Goal: Minimize the loss.

---

## 6. Accuracy (Classification)

Accuracy = Correct Predictions / Total Predictions

Example:

90 correct out of 100  

Accuracy = 90%

---

## 7. Gradient Descent (Basic Idea)

Gradient Descent is an optimization algorithm used to minimize the loss function.

Goal: Reduce error step by step.

---

### Why Needed?

When model makes predictions, it calculates error using a loss function.  
Gradient Descent helps to adjust model parameters (m and c in linear regression) to reduce that error.

---

### Basic Update Formula

m = m - learning_rate × derivative_of_loss  
c = c - learning_rate × derivative_of_loss  

Where:
- m = slope  
- c = intercept  
- learning_rate = small value (example: 0.01)  
- derivative = slope of error curve  

---

### Simple Understanding

1. Calculate prediction  
2. Calculate error  
3. Find slope (derivative)  
4. Update parameters  
5. Repeat until error becomes small  

---

### Visual Idea

Imagine you are standing on a mountain and want to reach the bottom.

- Look at slope  
- Take small step downward  
- Repeat  

Eventually you reach the minimum point.

---

### Learning Rate

Learning rate controls step size.

- Too small → Slow learning  
- Too large → May overshoot  
- Perfect value → Faster convergence  

---

### Goal of Gradient Descent

Minimize Loss Function  
Find best values of m and c  
Improve prediction accuracy
