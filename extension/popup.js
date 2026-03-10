document.addEventListener("DOMContentLoaded", () => {

  const userEmailEl = document.getElementById("user-email");

  // Get the user email dynamically
  chrome.identity.getProfileUserInfo((userInfo) => {
    if (userInfo && userInfo.email) {
      userEmailEl.textContent = userInfo.email;
    } else {
      userEmailEl.textContent = "Not signed in";
      userEmailEl.style.color = "red"; 
    }
  });

  // Load stored stats
  chrome.storage.local.get(["scanned", "scams"], (result) => {
    document.getElementById("scanned-count").textContent =
      result.scanned || 0;

    document.getElementById("scam-count").textContent =
      result.scams || 0;
  });

  // Add Scan Button Listener
  const scanButton = document.getElementById("scan-btn");

  if (scanButton) {
    scanButton.addEventListener("click", () => {

      chrome.runtime.sendMessage(
        { action: "scanEmails" },
        (response) => {

          if (response && !response.error) {
            document.getElementById("scanned-count").textContent =
              response.scanned;

            document.getElementById("scam-count").textContent =
              response.scams;
          } else {
            console.error("Scan failed:", response?.error);
          }
        }
      );

    });
  }

});