import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, Search, ShieldAlert, TriangleAlert, CheckCircle2, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

const fakeEmails = [
  {
    id: 1,
    from: "Amazon Billing <billing-security@amaz0n-secure-help.com>",
    subject: "Urgent: Verify your account immediately",
    preview: "Your account has been placed on hold. Click the secure link to verify...",
    body: "Hello customer,\n\nYour account has been placed on hold due to suspicious activity. You must verify your account immediately or it will be suspended. Click here now: http://amaz0n-secure-help.com/verify\n\nIf you need help, call 336-555-9012.\n\nRegards,\nAmazon Security",
    received: "9:18 AM"
  },
  {
    id: 2,
    from: "Professor Hall <ehall@college.edu>",
    subject: "Reminder about Thursday presentation",
    preview: "Bring your updated slides and be ready to present your prototype...",
    body: "Hi Cincear,\n\nJust a reminder to bring your updated slides for Thursday. Please be ready to present the prototype and your training results.\n\nBest,\nProfessor Hall",
    received: "8:02 AM"
  },
  {
    id: 3,
    from: "Bank Security Team <alerts@nationalbank-help.net>",
    subject: "Security alert: your account may be locked",
    preview: "We detected unusual activity. Act now to avoid permanent restrictions...",
    body: "Dear customer,\n\nWe detected unusual activity on your bank account. Act now to prevent your account from being locked. Confirm your password and SSN at www.nationalbank-help.net/security-check\n\nThank you,\nFraud Department",
    received: "Yesterday"
  },
  {
    id: 4,
    from: "Taylor <taylor@example.com>",
    subject: "Lunch tomorrow?",
    preview: "Still good for lunch at 1 tomorrow?",
    body: "Hey,\n\nAre you still good for lunch at 1 tomorrow? I can meet you by the student center.\n\n-Taylor",
    received: "Yesterday"
  }
];

const suspiciousTerms = [
  { term: "verify your account", reason: "This phrase is common in phishing campaigns.", weight: 0.18 },
  { term: "suspended", reason: "Threatening account suspension is a common pressure tactic.", weight: 0.15 },
  { term: "security alert", reason: "Authority-style warnings are often used to create panic.", weight: 0.14 },
  { term: "password", reason: "Requests for credentials are high risk.", weight: 0.2 },
  { term: "ssn", reason: "Requests for sensitive identity data are high risk.", weight: 0.22 },
  { term: "act now", reason: "Urgency language increases pressure on the recipient.", weight: 0.12 },
  { term: "click here", reason: "Direct call-to-click language is suspicious when tied to account actions.", weight: 0.11 },
  { term: "bank account", reason: "Financial account language can indicate fraud attempts.", weight: 0.13 },
  { term: "unusual activity", reason: "This phrase is often used to create fear and urgency.", weight: 0.1 },
  { term: "locked", reason: "Threat-based account language is suspicious.", weight: 0.1 }
];

const ruleChecks = [
  {
    label: "Contains a link",
    reason: "Scam messages often push users to fake login or payment pages.",
    test: /http[s]?:\/\/|www\./i,
    score: 0.35,
  },
  {
    label: "Contains a phone number",
    reason: "Scammers often try to move victims to phone calls they control.",
    test: /\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/i,
    score: 0.2,
  },
  {
    label: "Uses urgency language",
    reason: "Pressure language is a common social-engineering tactic.",
    test: /\burgent|immediately|act now|asap|final notice|suspended|locked\b/i,
    score: 0.25,
  },
  {
    label: "Requests sensitive information",
    reason: "Legitimate organizations do not ask for passwords or SSNs by email.",
    test: /\bpassword|ssn|bank account|verify your account|security alert\b/i,
    score: 0.35,
  },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function analyzeSelection(text) {
  const clean = text.toLowerCase();

  const ruleMatches = ruleChecks
    .filter((rule) => rule.test.test(text))
    .map((rule) => ({ label: rule.label, reason: rule.reason }));

  let ruleScore = 0;
  ruleChecks.forEach((rule) => {
    if (rule.test.test(text)) ruleScore += rule.score;
  });
  ruleScore = clamp(ruleScore, 0, 1);

  const phraseHits = suspiciousTerms
    .filter((item) => clean.includes(item.term))
    .map((item) => ({
      phrase: item.term,
      reason: item.reason,
      contribution: item.weight,
    }));

  const aiScore = clamp(
    0.12 +
      phraseHits.reduce((sum, item) => sum + item.contribution, 0) +
      (ruleMatches.length >= 2 ? 0.18 : 0) +
      (ruleMatches.length >= 3 ? 0.08 : 0),
    0,
    0.99
  );

  const riskScore = Number((aiScore * 0.7 + ruleScore * 0.3).toFixed(2));
  const classification = riskScore >= 0.5 ? "Scam" : "Safe";

  const advice =
    riskScore >= 0.7
      ? [
          "Do not click links or call numbers in the selected text.",
          "Verify the message through official contact information.",
          "Ask a trusted person before taking action."
        ]
      : riskScore >= 0.4
      ? ["Be cautious and verify the sender before responding."]
      : ["No strong scam indicators were detected in the selected text."];

  return {
    classification,
    riskScore,
    aiScore: Number(aiScore.toFixed(2)),
    ruleScore: Number(ruleScore.toFixed(2)),
    redFlags: ruleMatches,
    topPhrases: phraseHits.sort((a, b) => b.contribution - a.contribution).slice(0, 5),
    advice,
    reasons: ruleMatches.map((item) => item.label),
    fallbackUsed: true,
  };
}

async function scanWithBackend(text) {
  const response = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Backend error: ${response.status}`);
  }

  const data = await response.json();

  return {
    classification: data.classification,
    riskScore: data.risk_score,
    aiScore: data.ai_score,
    ruleScore: data.rule_score,
    redFlags: (data.reasons || []).map((reason) => ({
      label: reason,
      reason,
    })),
    topPhrases: (data.explanation?.top_phrases || []).map((item) => {
      if (typeof item === "string") {
        return { phrase: item, reason: "Model-highlighted phrase", contribution: 0 };
      }
      return {
        phrase: item.phrase || "Flagged phrase",
        reason: item.reason || "Model-highlighted phrase",
        contribution: item.contribution || 0,
      };
    }),
    advice: data.explanation?.what_to_do || ["No advice returned from backend."],
    reasons: data.reasons || [],
    fallbackUsed: false,
  };
}

function EmailBody({ text, onSelectionChange }) {
  const paragraphs = text.split("\n");

  return (
    <div
      className="rounded-2xl border bg-white p-5 text-[15px] text-slate-800 shadow-sm"
      onMouseUp={() => {
        const selected = window.getSelection()?.toString().trim() || "";
        onSelectionChange(selected);
      }}
    >
      {paragraphs.map((line, idx) => (
        <p key={idx} className="mb-4 whitespace-pre-wrap last:mb-0">
          {line || "\u00A0"}
        </p>
      ))}
    </div>
  );
}

export default function ElderGuardFakeOutlookDemo() {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(fakeEmails[0].id);
  const [selectedText, setSelectedText] = useState("");
  const [scanResult, setScanResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);

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
    } catch (error) {
      setScanResult(analyzeSelection(selectedText));
    } finally {
      setIsScanning(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <div className="border-b bg-white">
        <div className="mx-auto flex max-w-7xl items-center px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-slate-900 p-2 text-white">
              <Mail className="h-5 w-5" />
            </div>
            <h1 className="text-xl font-semibold">Outlook</h1>
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-7xl grid-cols-12 gap-4 px-6 py-6">
        <Card className="col-span-12 overflow-hidden rounded-2xl shadow-sm md:col-span-4">
          <CardHeader className="border-b pb-3">
            <CardTitle className="text-base">Inbox</CardTitle>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search mail"
                className="pl-9"
              />
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[76vh]">
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
                    className={`w-full border-b p-4 text-left transition ${
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
            </ScrollArea>
          </CardContent>
        </Card>

        <div className="col-span-12 md:col-span-8">
          <Card className="rounded-2xl shadow-sm">
            <CardHeader className="border-b">
              <div>
                <CardTitle className="text-lg">{selectedEmail.subject}</CardTitle>
                <p className="mt-1 text-sm text-slate-500">From: {selectedEmail.from}</p>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              <EmailBody text={selectedEmail.body} onSelectionChange={setSelectedText} />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Floating draggable widget button */}
      <motion.div
        drag
        dragMomentum={false}
        className="fixed bottom-6 right-6 z-50"
      >
        <Button
          onClick={runScan}
          disabled={!selectedText.trim() || isScanning}
          className="rounded-full h-14 w-14 bg-slate-900 hover:bg-slate-800 shadow-xl disabled:opacity-50"
          title={selectedText.trim() ? "Scan highlighted text" : "Highlight text first"}
        >
          <ShieldAlert className={`h-6 w-6 ${isScanning ? "animate-pulse" : ""}`} />
        </Button>
      </motion.div>

      <AnimatePresence>
        {scanResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed bottom-24 right-6 z-50 w-[420px] max-w-[calc(100vw-2rem)]"
          >
            <Card className="rounded-3xl border-slate-200 shadow-2xl">
              <CardHeader className="border-b pb-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <CardTitle className="text-lg">ElderGuard Scan</CardTitle>
                    {scanResult.fallbackUsed ? (
                      <p className="mt-1 text-xs text-amber-600">Backend unavailable. Showing local demo fallback.</p>
                    ) : (
                      <p className="mt-1 text-xs text-emerald-600">Connected to live Flask backend.</p>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-full"
                    onClick={() => setScanResult(null)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 p-5">
                <div className="flex items-center gap-2">
                  <Badge
                    className={`rounded-full px-3 py-1 ${
                      scanResult.classification === "Scam"
                        ? "bg-red-100 text-red-700"
                        : "bg-emerald-100 text-emerald-700"
                    }`}
                  >
                    {scanResult.classification === "Scam" ? <ShieldAlert className="mr-1 h-3.5 w-3.5" /> : <CheckCircle2 className="mr-1 h-3.5 w-3.5" />}
                    {scanResult.classification}
                  </Badge>
                  <Badge className="rounded-full bg-slate-100 text-slate-700">Risk {scanResult.riskScore}</Badge>
                </div>

                <div>
                  <h3 className="mb-2 text-sm font-semibold">Why it was flagged</h3>
                  {scanResult.redFlags.map((flag) => (
                    <div key={flag.label} className="text-sm text-slate-600">• {flag.label}</div>
                  ))}
                </div>

                <div>
                  <h3 className="mb-2 text-sm font-semibold">What to do</h3>
                  {scanResult.advice.map((line) => (
                    <div key={line} className="flex items-start gap-2 text-sm text-slate-600">
                      <TriangleAlert className="h-4 w-4 text-amber-500" />
                      <span>{line}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
