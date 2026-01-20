import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@shared/routes";
import { type SimulationRun } from "@shared/schema";

export function useSimulation() {
  const queryClient = useQueryClient();

  const latestRun = useQuery({
    queryKey: [api.simulation.getLatest.path],
    queryFn: async () => {
      const res = await fetch(api.simulation.getLatest.path);
      if (!res.ok) throw new Error("Failed to fetch status");
      const data = await res.json();
      return api.simulation.getLatest.responses[200].parse(data);
    },
    // Poll every second while running, otherwise every 5s
    refetchInterval: (query) => {
      const data = query.state.data as SimulationRun | null | undefined;
      return data?.status === "running" || data?.status === "pending" ? 1000 : 5000;
    },
  });

  const startSimulation = useMutation({
    mutationFn: async () => {
      const res = await fetch(api.simulation.start.path, {
        method: api.simulation.start.method,
      });
      if (!res.ok) throw new Error("Failed to start simulation");
      return api.simulation.start.responses[201].parse(await res.json());
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [api.simulation.getLatest.path] });
    },
  });

  return {
    latestRun,
    startSimulation,
  };
}
