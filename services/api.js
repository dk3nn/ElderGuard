const API_BASE_URL = "http://localhost:8000"; 
// later becomes deployed URL

export async function analyzeMessage(text) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    throw new Error("API request failed");
  }

  return response.json();
}