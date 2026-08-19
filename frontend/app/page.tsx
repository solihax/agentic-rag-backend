"use client";

import { useState } from "react";

interface SourceInfo {
  filename: string;
  source_type: string;
}

interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center px-4 py-10">
      <div className="w-full max-w-2xl">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">
          Agentic RAG Assistant
        </h1>
        <p className="text-sm text-gray-500 mb-6">
          Ask a question about your ingested documents.
        </p>

        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            placeholder="Ask a question..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 mb-4">
            {error}
          </div>
        )}

        {answer && (
          <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-500 mb-2 font-medium">Answer</p>
            <p className="text-gray-800 whitespace-pre-wrap">{answer}</p>
          </div>
        )}

        {sources.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <p className="text-sm text-gray-500 mb-2 font-medium">Sources</p>
            <ul className="text-sm text-gray-700 space-y-1">
              {sources.map((s, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="inline-block px-2 py-0.5 bg-gray-100 rounded text-xs uppercase text-gray-500">
                    {s.source_type}
                  </span>
                  {s.filename}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  );
}
