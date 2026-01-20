import { db } from "./db";
import { simulationRuns, type SimulationRun, type InsertSimulationRun } from "@shared/schema";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  createSimulationRun(run: InsertSimulationRun): Promise<SimulationRun>;
  updateSimulationRun(id: number, updates: Partial<SimulationRun>): Promise<SimulationRun>;
  getLatestSimulationRun(): Promise<SimulationRun | undefined>;
}

export class DatabaseStorage implements IStorage {
  async createSimulationRun(run: InsertSimulationRun): Promise<SimulationRun> {
    const [newRun] = await db.insert(simulationRuns).values(run).returning();
    return newRun;
  }

  async updateSimulationRun(id: number, updates: Partial<SimulationRun>): Promise<SimulationRun> {
    const [updated] = await db.update(simulationRuns)
      .set(updates)
      .where(eq(simulationRuns.id, id))
      .returning();
    return updated;
  }

  async getLatestSimulationRun(): Promise<SimulationRun | undefined> {
    const [run] = await db.select()
      .from(simulationRuns)
      .orderBy(desc(simulationRuns.id))
      .limit(1);
    return run;
  }
}

export const storage = new DatabaseStorage();
