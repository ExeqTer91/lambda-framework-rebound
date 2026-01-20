import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";

interface SimulationCardProps {
  title: string;
  description: string;
  isRunning: boolean;
  onRun: () => void;
  className?: string;
}

export function SimulationCard({ title, description, isRunning, onRun, className }: SimulationCardProps) {
  return (
    <Card className={cn("glass-panel glow-border border-primary/20", className)}>
      <CardHeader>
        <CardTitle className="text-2xl md:text-3xl text-primary font-display tracking-tight">
          {title}
        </CardTitle>
        <CardDescription className="text-base text-muted-foreground">
          {description}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-6">
           <div className="p-3 rounded-lg bg-secondary/50 border border-white/5">
             <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Integrator</div>
             <div className="text-sm font-medium font-mono">WHFast (Symplectic)</div>
           </div>
           <div className="p-3 rounded-lg bg-secondary/50 border border-white/5">
             <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Timestep</div>
             <div className="text-sm font-medium font-mono">5% Inner Orbit</div>
           </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          onClick={onRun} 
          disabled={isRunning}
          size="lg"
          className="w-full font-semibold shadow-lg shadow-primary/20 hover:shadow-primary/40 transition-all"
        >
          {isRunning ? (
            <>
              <RotateCw className="mr-2 h-4 w-4 animate-spin" />
              Simulation Running...
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4 fill-current" />
              Start New Simulation
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
