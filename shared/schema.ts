import { pgTable, text, serial, timestamp, boolean, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const simulationRuns = pgTable("simulation_runs", {
  id: serial("id").primaryKey(),
  status: text("status").notNull().default("pending"), // pending, running, completed, failed
  output: text("output").default(""),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertSimulationRunSchema = createInsertSchema(simulationRuns).omit({
  id: true,
  createdAt: true,
  output: true,
  status: true
});

export type SimulationRun = typeof simulationRuns.$inferSelect;
export type InsertSimulationRun = z.infer<typeof insertSimulationRunSchema>;
