document.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("https://your-backend-api.com/scan-status");
    const data = await response.json();

    document.getElementById("scanned-count").textContent = data.scanned;
    document.getElementById("scam-count").textContent = data.scams;

  } catch (error) {
    console.error("Error fetching scan data:", error);
  }
});