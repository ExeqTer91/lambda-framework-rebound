import { z } from 'zod';
import { simulationRuns } from './schema';

export const api = {
  simulation: {
    start: {
      method: 'POST' as const,
      path: '/api/simulation/start',
      responses: {
        201: z.custom<typeof simulationRuns.$inferSelect>(),
      },
    },
    getLatest: {
      method: 'GET' as const,
      path: '/api/simulation/latest',
      responses: {
        200: z.custom<typeof simulationRuns.$inferSelect>().nullable(),
      },
    },
  },
};
