import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useHandTracking } from "@/hooks/useHandTracking";
import { useSentenceBuilder } from "@/hooks/useSentenceBuilder";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "SignSpeak — Real-time Sign Language Translator" },
      {
        name: "description",
        content:
          "Browser-based sign language to speech translator using MediaPipe hand tracking and an LSTM model running on TensorFlow.js.",
      },
    ],
  }),
});

function Index() {
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const sentenceBuilder = useSentenceBuilder({ ttsEnabled });
  const { addPrediction } = sentenceBuilder;

  const onPrediction = useCallback(
    (p: { word: string; confidence: number }) => addPrediction(p.word, p.confidence),
    [addPrediction],
  );

  const { videoRef, canvasRef, prediction, isReady, demoMode, status, handVisible } = useHandTracking({ onPrediction });
  const hasSentence = useMemo(() => sentenceBuilder.words.length > 0, [sentenceBuilder.words.length]);

  return (
    <main className="min-h-screen bg-[radial-gradient(80%_50%_at_10%_0%,rgba(130,90,255,0.16),transparent),radial-gradient(70%_45%_at_100%_0%,rgba(70,210,180,0.14),transparent)] bg-background text-foreground">
      <header className="border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <div
              className="h-8 w-8 rounded-lg"
              style={{ background: "var(--gradient-primary)" }}
            />
            <span className="font-semibold tracking-tight">SignSpeak</span>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <a href="#model-setup" className="text-muted-foreground hover:text-foreground">
              Model Setup
            </a>
            <Link to="/about" className="text-muted-foreground hover:text-foreground">
              About
            </Link>
            <a
              href="https://github.com"
              className="text-muted-foreground hover:text-foreground"
            >
              Docs
            </a>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 pt-10 pb-6">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Sign language → spoken English, live in your browser.
        </h1>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          MediaPipe extracts 21 hand landmarks per frame. A 30-frame sliding
          buffer feeds an LSTM that recognizes signs, then a pause-detection
          sentence builder triggers the Web Speech API.
        </p>

        {demoMode && (
          <div id="model-setup" className="mt-5 rounded-xl border border-border bg-card/80 p-4 text-sm shadow-sm backdrop-blur">
            <strong>Model setup needed for real recognition.</strong>{" "}
            This app currently runs in demo visualization mode. For actual sign recognition, place a pre-trained TF.js model in{" "}
            <code>public/model/</code> and labels in <code>public/labels.json</code>. Then reload.
          </div>
        )}
      </section>

      <section className="mx-auto grid max-w-6xl gap-6 px-6 pb-10 lg:grid-cols-[1fr_380px]">
        {/* Webcam stage */}
        <div
          className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-border bg-black"
          style={{ boxShadow: "var(--shadow-elegant)" }}
        >
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
            autoPlay
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 h-full w-full"
          />

          {/* Status pill */}
          <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs text-white backdrop-blur">
            <span
              className={`h-2 w-2 rounded-full ${
                isReady ? (handVisible ? "bg-emerald-400" : "bg-amber-400") : "bg-zinc-400"
              }`}
            />
            {status}
          </div>

          {/* Live prediction */}
          {prediction.word && (
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-white/20 bg-white/95 px-5 py-2 text-lg font-medium text-foreground shadow-lg">
              {prediction.word}
              <span className="ml-2 text-sm text-muted-foreground">
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
          )}
        </div>

        {/* Side panel */}
        <aside className="flex flex-col gap-4">
          <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">
              Current sentence
            </p>
            <p className="mt-2 min-h-[3rem] text-lg leading-relaxed">
              {sentenceBuilder.words.length > 0 ? (
                sentenceBuilder.words.join(" ")
              ) : (
                <span className="text-muted-foreground">Start signing…</span>
              )}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={sentenceBuilder.speakNow}
                disabled={!hasSentence}
                className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                style={{ background: "var(--gradient-primary)" }}
              >
                Speak
              </button>
              <button
                onClick={() => setTtsEnabled((v) => !v)}
                className="cursor-pointer rounded-lg border border-border px-4 py-2 text-sm hover:bg-accent active:scale-[0.98]"
              >
                Auto-TTS: {ttsEnabled ? "ON" : "OFF"}
              </button>
              <button
                onClick={sentenceBuilder.clear}
                className="cursor-pointer rounded-lg border border-border px-4 py-2 text-sm hover:bg-accent active:scale-[0.98]"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">
              Session history
            </p>
            {sentenceBuilder.history.length === 0 ? (
              <p className="mt-2 text-sm text-muted-foreground">
                Completed sentences appear here.
              </p>
            ) : (
              <ul className="mt-2 divide-y divide-border">
                {sentenceBuilder.history.map((h, i) => (
                  <li
                    key={i}
                    className="flex items-center justify-between gap-3 py-2 text-sm"
                  >
                    <span className="truncate">{h.text}</span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {h.time}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <h2 className="text-xl font-semibold">Pipeline</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {[
            ["Webcam", "getUserMedia"],
            ["MediaPipe", "21 landmarks/hand"],
            ["Buffer", "30-frame window"],
            ["LSTM", "TF.js inference"],
            ["Sentence", "1.5s pause = end"],
            ["TTS", "Web Speech API"],
          ].map(([title, sub]) => (
            <div key={title} className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-medium">{title}</p>
              <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
