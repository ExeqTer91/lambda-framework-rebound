import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Ensure public directory exists for results
  const publicDir = path.join(process.cwd(), "client", "public");
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  app.post(api.simulation.start.path, async (req, res) => {
    try {
      // Create a new run record
      const run = await storage.createSimulationRun({
        status: "running",
        output: "Initializing simulation...\n",
      });

      // Spawn python process
      // Use stdbuf to unbuffer output if possible, or python -u
      const pythonProcess = spawn("python3", ["-u", "server/simulation.py"], {
        cwd: process.cwd(),
      });

      let outputBuffer = "Initializing simulation...\n";

      const updateOutput = async () => {
        try {
          await storage.updateSimulationRun(run.id, { output: outputBuffer });
        } catch (e) {
          console.error("Failed to update output log", e);
        }
      };

      // Debounced update (every 500ms) to avoid spamming DB
      let updateTimeout: NodeJS.Timeout | null = null;
      const scheduleUpdate = () => {
        if (!updateTimeout) {
          updateTimeout = setTimeout(() => {
            updateOutput();
            updateTimeout = null;
          }, 500);
        }
      };

      pythonProcess.stdout.on("data", (data) => {
        const chunk = data.toString();
        outputBuffer += chunk;
        scheduleUpdate();
      });

      pythonProcess.stderr.on("data", (data) => {
        const chunk = data.toString();
        outputBuffer += `[ERROR] ${chunk}`;
        scheduleUpdate();
      });

      pythonProcess.on("close", async (code) => {
        outputBuffer += `\nProcess exited with code ${code}`;
        const status = code === 0 ? "completed" : "failed";
        
        if (updateTimeout) clearTimeout(updateTimeout);
        
        await storage.updateSimulationRun(run.id, {
          status: status,
          output: outputBuffer
        });
      });

      res.status(201).json(run);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Failed to start simulation" });
    }
  });

  app.get(api.simulation.getLatest.path, async (req, res) => {
    const run = await storage.getLatestSimulationRun();
    res.json(run || null);
  });

  return httpServer;
}
