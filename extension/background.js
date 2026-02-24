chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "login") {
    chrome.identity.getAuthToken({ interactive: true }, function (token) {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError);
        return;
      }

      console.log("Access Token:", token);

      chrome.storage.local.set({ accessToken: token });
      sendResponse({ success: true, token: token });
    });

    return true; // Required for async response
  }
});
