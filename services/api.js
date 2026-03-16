const API_BASE_URL = "http://127.0.0.1:5000";

export async function analyzeMessage(message) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });

  if (!response.ok) {
    throw new Error("API request failed");
  }

  return response.json();
}