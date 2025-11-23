# **Movie Recommendation System (Content Similarity Based)**

This project is a Python-based Content Similarity Movie Recommendation System developed for academic submission.
It recommends the **Top 10 most similar movies** to any selected movie using a combination of genre similarity, language matching, and numerical feature similarity (cosine similarity).

---

## **Project Features**

### **1. Full Data Cleaning Pipeline**
- Handles missing values
- Fixes inconsistent or invalid date formats
- Normalizes numerical fields
- Processes multi-valued genre and language fields

---

### **2. Feature Scaling with Min-Max Normalization**
Applied to:

- Popularity
- Budget
- Revenue
- Runtime
- Vote Average
- Vote Count
- Release Year

---

### **3. Genre and Language Similarity**
- **Genre Similarity:** Computed using custom Jaccard similarity
- **Language Similarity:** Exact-match scoring

---

### **4. Numerical Similarity Using Cosine Similarity**
- Extracts numerical feature vectors
- Computes cosine similarity between movies based on numerical attributes

---

### **5. Combined Similarity Scoring Algorithm**

Final Score =  
50% Genre Similarity  
20% Language Similarity  
30% Numerical Feature Similarity (Cosine)

This weighted scoring ensures balanced recommendations across content and numerical features.

---

### **6. Top 10 Movie Recommendations**
- Returns the ten closest movies based on the combined similarity score
- Sorted from highest to lowest similarity

---

## **Evaluation Metrics Implemented**

- **Precision@10** – Accuracy of recommended movies
- **Mean Jaccard Similarity** – Average genre overlap
- **Language Match Rate** – Percentage of recommendations with the same language
- **Average Similarity Score** – Mean final similarity score
- **Response Time Measurement** – Time taken to generate recommendations

---

## **Data Visualization Suite**

This project generates several analytical charts:

- Language distribution pie chart
- Genre frequency bar chart
- Popularity vs. rating scatter plot
- Movies-per-year trend graph
- Budget vs. revenue (ROI) scatter plot
- Votes vs. rating plot
- Performance metrics bar graph

---

## **Interactive CLI Program**

A command-line tool that allows users to:

- Enter any movie name
- Receive the Top 10 recommended movies
- View similarity scores

---

## **Dataset and Report**

- Includes a fully cleaned and processed dataset
- Includes a complete PDF project report for academic submission

---

## **Project Structure**

Movie-Recommendation-System/
│── README.md  
│── movie_recommender.py  
│── data_cleaning.py  
│── similarity_metrics.py  
│── metrics.py  
│── visualizations.py  
│── cli_app.py  
│── report.pdf  
│── data/  
│     └── movies_cleaned.csv  

---

## **Technologies Used**

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Cosine Similarity
- Jaccard Similarity
