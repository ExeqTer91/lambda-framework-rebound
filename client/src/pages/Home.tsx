import { useSimulation } from "@/hooks/use-simulation";
import { TerminalOutput } from "@/components/TerminalOutput";
import { SimulationCard } from "@/components/SimulationCard";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, LineChart, Cpu, Activity } from "lucide-react";

export default function Home() {
  const { latestRun, startSimulation } = useSimulation();

  const run = latestRun.data;
  const isRunning = run?.status === "running" || run?.status === "pending";
  const isCompleted = run?.status === "completed";

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-[#0a0f1c] to-[#050810] text-foreground p-4 md:p-8 lg:p-12">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-white/10 pb-8">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold font-display bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              λ-Framework
            </h1>
            <p className="mt-2 text-lg text-muted-foreground max-w-2xl">
              N-Body Stability Simulator powered by REBOUND
            </p>
          </div>
          <div className="flex gap-4">
             <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium">
               <Cpu className="w-4 h-4" />
               <span>v3.11.0 Core</span>
             </div>
             <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 text-green-500 text-sm font-medium">
               <Activity className="w-4 h-4" />
               <span>Active</span>
             </div>
          </div>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 min-h-[600px]">
          
          {/* Left Column: Controls & Info */}
          <div className="lg:col-span-4 space-y-6">
            <SimulationCard 
              title="Stability Test"
              description="Validates planetary system stability using λ = √φ ≈ 1.272 spacing vs random configurations."
              isRunning={isRunning}
              onRun={() => startSimulation.mutate()}
            />

            {startSimulation.isError && (
              <Alert variant="destructive" className="bg-destructive/10 border-destructive/20 text-destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                  {startSimulation.error instanceof Error ? startSimulation.error.message : "Failed to start simulation"}
                </AlertDescription>
              </Alert>
            )}

            <div className="glass-panel p-6 rounded-xl">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white/90">
                <LineChart className="w-5 h-5 text-primary" />
                <span>Simulation Specs</span>
              </h3>
              <ul className="space-y-3 text-sm text-muted-foreground">
                <li className="flex justify-between">
                  <span>Golden Ratio (φ)</span>
                  <span className="font-mono text-primary">1.6180339887</span>
                </li>
                <li className="flex justify-between">
                  <span>Lambda (λ = √φ)</span>
                  <span className="font-mono text-primary">1.2720196495</span>
                </li>
                <li className="flex justify-between">
                  <span>Planets</span>
                  <span className="font-mono text-white">4</span>
                </li>
                <li className="flex justify-between">
                  <span>Mass</span>
                  <span className="font-mono text-white">3e-6 M☉</span>
                </li>
                <li className="flex justify-between">
                  <span>Integration</span>
                  <span className="font-mono text-white">1e5 Periods</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Right Column: Output & Visuals */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            
            {/* Terminal Output */}
            <div className="flex-1 min-h-[400px]">
              <TerminalOutput 
                output={run?.output || ""} 
                status={run?.status || "pending"} 
                className="h-full min-h-[400px]"
              />
            </div>

            {/* Results Visualization */}
            {isCompleted && (
              <div className="glass-panel p-1 rounded-xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className="relative aspect-video bg-black/50 rounded-lg overflow-hidden border border-white/5">
                  <img 
                    src={`/results.png?t=${run?.id || Date.now()}`}
                    alt="Simulation Results" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-4 right-4 px-3 py-1 bg-black/80 backdrop-blur text-xs font-mono text-white/70 rounded border border-white/10">
                    Generated Plot
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
