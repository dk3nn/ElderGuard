chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

  //  LOGIN
  if (request.action === "login") {
    chrome.identity.getAuthToken({ interactive: true }, function (token) {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError);
        return;
      }

      chrome.storage.local.set({ accessToken: token });
      sendResponse({ success: true });
    });

    return true;
  }

  // SCAN EMAILS
  if (request.action === "scanEmails") {
    chrome.storage.local.get(["accessToken"], async (result) => {
      const token = result.accessToken;

      if (!token) {
        sendResponse({ error: "Not logged in" });
        return;
      }

      try {
        //  Get recent messages
        const messageList = await fetch(
          "https://www.googleapis.com/gmail/v1/users/me/messages?maxResults=5",
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );

        const messageData = await messageList.json();

        let scanned = 0;
        let scams = 0;

        //  Loop through emails
        for (let msg of messageData.messages) {
          const fullMessage = await fetch(
            `https://www.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=full`,
            {
              headers: { Authorization: `Bearer ${token}` }
            }
          );

          const email = await fullMessage.json();

          //  Extract body text
          let body = "";

          if (email.payload.parts) {
            const part = email.payload.parts.find(
              p => p.mimeType === "text/plain"
            );

            if (part && part.body.data) {
              body = atob(part.body.data.replace(/-/g, "+").replace(/_/g, "/"));
            }
          }

          if (!body) continue;

          scanned++;

          //  Send to ML Backend
          const mlResponse = await fetch("http://localhost:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: body })
          });

          const mlData = await mlResponse.json();

          if (mlData.prediction === "scam") {
            scams++;
          }
        }

        //  Store results
        chrome.storage.local.set({ scanned, scams });

        sendResponse({ scanned, scams });

      } catch (error) {
        console.error(error);
        sendResponse({ error: "Scan failed" });
      }
    });

    return true;
  }
});