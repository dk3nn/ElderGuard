import React, { useMemo, useState } from "react";
import { Mail, Search, ShieldAlert, TriangleAlert, CheckCircle2, X } from "lucide-react";

const fakeEmails = [
  {
    id: 1,
    from: "Amazon Billing <billing-security@amaz0n-secure-help.com>",
    subject: "Urgent: Verify your account immediately",
    preview: "Your account has been placed on hold. Click the secure link to verify...",
    body: "Hello customer,\n\nYour account has been placed on hold due to suspicious activity. You must verify your account immediately or it will be suspended. Click here now: http://amaz0n-secure-help.com/verify\n\nIf you need help, call 336-555-9012.\n\nRegards,\nAmazon Security",
    received: "9:18 AM",
  },
  {
    id: 2,
    from: "Professor Hall <ehall@college.edu>",
    subject: "Reminder about Thursday presentation",
    preview: "Bring your updated slides and be ready to present your prototype...",
    body: "Hi Cincear,\n\nJust a reminder to bring your updated slides for Thursday. Please be ready to present the prototype and your training results.\n\nBest,\nProfessor Hall",
    received: "8:02 AM",
  },
  {
    id: 3,
    from: "Bank Security Team <alerts@nationalbank-help.net>",
    subject: "Security alert: your account may be locked",
    preview: "We detected unusual activity. Act now to avoid permanent restrictions...",
    body: "Dear customer,\n\nWe detected unusual activity on your bank account. Act now to prevent your account from being locked. Confirm your password and SSN at www.nationalbank-help.net/security-check\n\nThank you,\nFraud Department",
    received: "Yesterday",
  },
  {
    id: 4,
    from: "Taylor <taylor@example.com>",
    subject: "Lunch tomorrow?",
    preview: "Still good for lunch at 1 tomorrow?",
    body: "Hey,\n\nAre you still good for lunch at 1 tomorrow? I can meet you by the student center.\n\n-Taylor",
    received: "Yesterday",
  },
];

function fallbackAnalyze(text) {
  const lower = text.toLowerCase();
  let ruleScore = 0;
  const redFlags = [];

  if (/http[s]?:\/\/|www\./i.test(text)) {
    ruleScore += 0.4;
    redFlags.push({
      label: "Contains a link",
      reason: "Scam messages often push users to fake websites.",
    });
  }

  if (/\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/i.test(text)) {
    ruleScore += 0.25;
    redFlags.push({
      label: "Contains a phone number",
      reason: "Scammers often try to move people to phone calls they control.",
    });
  }

  if (/\burgent|immediately|act now|asap|suspended|locked\b/i.test(text)) {
    ruleScore += 0.25;
    redFlags.push({
      label: "Uses urgency language",
      reason: "Pressure language is a common scam tactic.",
    });
  }

  if (/\bpassword|ssn|bank|account|verify\b/i.test(text)) {
    ruleScore += 0.3;
    redFlags.push({
      label: "Requests sensitive information",
      reason: "Requests for private credentials or account details are risky.",
    });
  }

  ruleScore = Math.min(ruleScore, 1);

  const topPhrases = [
    "verify your account",
    "urgent",
    "bank account",
    "password",
    "security alert",
  ]
    .filter((phrase) => lower.includes(phrase))
    .map((phrase) => ({
      phrase,
      contribution: 0.12,
      reason: "This phrase commonly appears in phishing-style messages.",
    }));

  const aiScore = Math.min(0.15 + topPhrases.length * 0.16 + (redFlags.length >= 2 ? 0.18 : 0), 0.99);
  const riskScore = Number((aiScore * 0.7 + ruleScore * 0.3).toFixed(2));

  return {
    classification: riskScore >= 0.5 ? "Scam" : "Safe",
    riskScore,
    aiScore: Number(aiScore.toFixed(2)),
    ruleScore: Number(ruleScore.toFixed(2)),
    redFlags,
    topPhrases,
    advice:
      riskScore >= 0.7
        ? [
            "Do not click links or call numbers in the selected text.",
            "Verify through official contact information.",
            "Ask a trusted person before taking action.",
          ]
        : riskScore >= 0.4
        ? ["Be cautious and verify the sender before responding."]
        : ["No strong scam indicators were detected in the selected text."],
    fallbackUsed: true,
  };
}

async function scanWithBackend(text) {
  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }

  const data = await res.json();

  return {
    classification: data.classification,
    riskScore: data.risk_score,
    aiScore: data.ai_score,
    ruleScore: data.rule_score,
    redFlags: (data.reasons || []).map((reason) => ({
      label: reason,
      reason,
    })),
    topPhrases: (data.explanation?.top_phrases || []).map((item) =>
      typeof item === "string"
        ? { phrase: item, contribution: 0, reason: "Model-highlighted phrase" }
        : {
            phrase: item.phrase || "Flagged phrase",
            contribution: item.contribution || 0,
            reason: item.reason || "Model-highlighted phrase",
          }
    ),
    advice: data.explanation?.what_to_do || ["No advice returned from backend."],
    fallbackUsed: false,
  };
}

function EmailBody({ text, onSelectionChange }) {
  return (
    <div
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      onMouseUp={() => {
        const selected = window.getSelection()?.toString().trim() || "";
        onSelectionChange(selected);
      }}
    >
      {text.split("\n").map((line, idx) => (
        <p key={idx} className="mb-4 whitespace-pre-wrap text-[15px] leading-7 text-slate-800 last:mb-0">
          {line || "\u00A0"}
        </p>
      ))}
    </div>
  );
}

export default function AppDemo() {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(fakeEmails[0].id);
  const [selectedText, setSelectedText] = useState("");
  const [scanResult, setScanResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [widgetPos, setWidgetPos] = useState({ x: window.innerWidth - 110, y: window.innerHeight - 110 });
  const [dragging, setDragging] = useState(false);

  const filteredEmails = useMemo(() => {
    const q = search.toLowerCase().trim();
    if (!q) return fakeEmails;
    return fakeEmails.filter((email) =>
      `${email.from} ${email.subject} ${email.preview}`.toLowerCase().includes(q)
    );
  }, [search]);

  const selectedEmail = filteredEmails.find((email) => email.id === selectedId) || fakeEmails[0];

  async function runScan() {
    if (!selectedText.trim()) return;

    setIsScanning(true);
    try {
      const result = await scanWithBackend(selectedText);
      setScanResult(result);
    } catch {
      setScanResult(fallbackAnalyze(selectedText));
    } finally {
      setIsScanning(false);
    }
  }

  function startDrag(e) {
    setDragging(true);

    const move = (event) => {
      setWidgetPos({
        x: event.clientX - 28,
        y: event.clientY - 28,
      });
    };

    const stop = () => {
      setDragging(false);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", stop);
    };

    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        backgroundColor: "#f1f5f9",
        color: "#0f172a",
        margin: 0,
        padding: 0,
      }}
    >
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-slate-900 p-2 text-white">
              <Mail className="h-5 w-5" />
            </div>
            <h1 className="text-xl font-semibold text-slate-900">Outlook</h1>
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-7xl grid-cols-12 gap-6 px-6 py-6">
        <div className="col-span-12 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm md:col-span-4">
          <div className="border-b border-slate-200 p-4">
            <h2 className="text-base font-semibold text-slate-900">Inbox</h2>
            <div className="relative mt-3">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search mail"
                className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm outline-none ring-0 placeholder:text-slate-400 focus:border-slate-300"
              />
            </div>
          </div>

          <div className="h-[75vh] overflow-y-auto">
            {filteredEmails.map((email) => {
              const active = email.id === selectedEmail.id;
              return (
                <button
                  key={email.id}
                  type="button"
                  onClick={() => {
                    setSelectedId(email.id);
                    setSelectedText("");
                    setScanResult(null);
                  }}
                  className={`w-full border-b border-slate-200 p-4 text-left transition ${
                    active ? "bg-slate-900 text-white" : "bg-white hover:bg-slate-50"
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between gap-3">
                    <span className={`truncate text-sm font-medium ${active ? "text-white" : "text-slate-800"}`}>
                      {email.from}
                    </span>
                    <span className={`shrink-0 text-xs ${active ? "text-slate-300" : "text-slate-400"}`}>
                      {email.received}
                    </span>
                  </div>
                  <div className={`truncate text-sm font-semibold ${active ? "text-white" : "text-slate-900"}`}>
                    {email.subject}
                  </div>
                  <div className={`mt-1 truncate text-sm ${active ? "text-slate-300" : "text-slate-500"}`}>
                    {email.preview}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="col-span-12 md:col-span-8">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-900">{selectedEmail.subject}</h2>
              <p className="mt-1 text-sm text-slate-500">From: {selectedEmail.from}</p>
            </div>
            <div className="p-6">
              <EmailBody text={selectedEmail.body} onSelectionChange={setSelectedText} />
            </div>
          </div>
        </div>
      </div>

      <div
        onMouseDown={startDrag}
        style={{
          position: "fixed",
          left: widgetPos.x,
          top: widgetPos.y,
          zIndex: 1000,
          cursor: dragging ? "grabbing" : "grab",
        }}
      >
        <button
          onClick={(e) => {
            e.stopPropagation();
            runScan();
          }}
          disabled={!selectedText.trim() || isScanning}
          title={selectedText.trim() ? "Scan highlighted text" : "Highlight text first"}
          className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-900 text-white shadow-xl transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ShieldAlert className={`h-6 w-6 ${isScanning ? "animate-pulse" : ""}`} />
        </button>
      </div>

      {scanResult && (
        <div className="fixed bottom-24 right-6 z-[1001] w-[420px] max-w-[calc(100vw-2rem)]">
          <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-2xl">
            <div className="border-b border-slate-200 p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">ElderGuard Scan</h3>
                  {scanResult.fallbackUsed ? (
                    <p className="mt-1 text-xs text-amber-600">Backend unavailable. Showing local demo fallback.</p>
                  ) : (
                    <p className="mt-1 text-xs text-emerald-600">Connected to live Flask backend.</p>
                  )}
                </div>
                <button
                  onClick={() => setScanResult(null)}
                  className="rounded-full p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="space-y-4 p-5">
              <div className="flex items-center gap-2">
                <span
                  className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${
                    scanResult.classification === "Scam"
                      ? "bg-red-100 text-red-700"
                      : "bg-emerald-100 text-emerald-700"
                  }`}
                >
                  {scanResult.classification === "Scam" ? (
                    <ShieldAlert className="mr-1 h-3.5 w-3.5" />
                  ) : (
                    <CheckCircle2 className="mr-1 h-3.5 w-3.5" />
                  )}
                  {scanResult.classification}
                </span>

                <span className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-700">
                  Risk {scanResult.riskScore}
                </span>
              </div>

              <div>
                <h4 className="mb-2 text-sm font-semibold text-slate-900">Why it was flagged</h4>
                {scanResult.redFlags.length ? (
                  scanResult.redFlags.map((flag) => (
                    <div key={flag.label} className="mb-2 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
                      <div className="font-medium text-slate-900">{flag.label}</div>
                      <div className="mt-1 text-slate-500">{flag.reason}</div>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-slate-500">No rule-based red flags matched.</div>
                )}
              </div>

              <div>
                <h4 className="mb-2 text-sm font-semibold text-slate-900">What to do</h4>
                {scanResult.advice.map((line) => (
                  <div key={line} className="mb-2 flex items-start gap-2 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
                    <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                    <span>{line}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
