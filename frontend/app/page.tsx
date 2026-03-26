"use client";

import React, { useState, FormEvent } from 'react';
import './globals.css';

interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  year: number;
  venue: string;
  url: string;
  rank: number;
  relevance_score: number;
  combined_score: number;
}

interface ApiResponse {
  status: string;
  saved_id: number;
  recommendations: Paper[];
  query_embeddings: number[];
  summary_embeddings_shape: number[];
}

export default function Page() {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        `http://localhost:8000/api/search/?query=${encodeURIComponent(searchQuery)}`
      );

      const data = await response.json();

      if (response.ok) {
        setResult(data);
        setSearchQuery("");
      } else {
        setError(data.error || "Something went wrong.");
      }
    } catch (error) {
      setError("Could not connect to the server.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="container">
      <div className="upload-card">
        <h1 className="title">Recommendation System</h1>

        <form onSubmit={handleSubmit} className="search-wrapper">
          <input
            type="text"
            placeholder="Type your query here..."
            className="search-input-unique"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button
            type="submit"
            className="submit-btn-unique"
            disabled={isSubmitting}
          >
            {isSubmitting ? "..." : "Search"}
          </button>
        </form>

        {error && <p style={{ color: "red" }}>{error}</p>}

        {result && (
          <div>
            {result.recommendations.map((paper) => (
              <div key={paper.id} style={{ marginBottom: "1.5rem" }}>
                <h2>#{paper.rank} — <a href={paper.url}>{paper.title}</a></h2>
                <p>{paper.authors.join(", ")}</p>
                <p>{paper.abstract}</p>
                <p>Year: {paper.year} · Venue: {paper.venue} · Score: {paper.relevance_score}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}