import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Terminal, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface TerminalOutputProps {
  output: string;
  status: "pending" | "running" | "completed" | "failed";
  className?: string;
}

export function TerminalOutput({ output, status, className }: TerminalOutputProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when output changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [output]);

  return (
    <div className={cn("flex flex-col h-full overflow-hidden rounded-xl border border-border bg-black/90 font-mono text-sm shadow-2xl", className)}>
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-white/5 border-b border-white/5">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Terminal className="w-4 h-4" />
          <span className="text-xs font-medium uppercase tracking-wider">System Log</span>
        </div>
        <div className="flex items-center gap-2">
          {status === "running" && (
            <span className="flex items-center gap-1.5 text-xs text-primary animate-pulse">
              <Loader2 className="w-3 h-3 animate-spin" />
              PROCESSING
            </span>
          )}
          {status === "completed" && (
            <span className="flex items-center gap-1.5 text-xs text-green-500">
              <CheckCircle2 className="w-3 h-3" />
              DONE
            </span>
          )}
          {status === "failed" && (
            <span className="flex items-center gap-1.5 text-xs text-destructive">
              <AlertCircle className="w-3 h-3" />
              ERROR
            </span>
          )}
        </div>
      </div>

      {/* Terminal Content */}
      <div className="flex-1 overflow-auto p-4 custom-scrollbar">
        <pre className="font-mono text-xs md:text-sm leading-relaxed whitespace-pre-wrap break-all text-primary/90 terminal-text">
          {output || "// Waiting for process to start..."}
          <div ref={bottomRef} />
        </pre>
      </div>
    </div>
  );
}
