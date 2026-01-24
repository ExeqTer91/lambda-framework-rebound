# λ-Framework Rebound Validation

## Overview

This project validates the λ-Framework hypothesis using REBOUND N-body simulations, testing whether planetary systems with λ = √φ ≈ 1.272 spacing provide enhanced dynamical stability. The repository includes a full-stack web application for running and visualizing simulations, along with supporting experiments for AI linguistics research (Trinity Architecture).

The core scientific question: Do planetary systems spaced by the golden ratio's square root exhibit stability comparable to exact mean-motion resonances?

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Routing**: Wouter (lightweight alternative to React Router)
- **State Management**: TanStack Query for server state
- **Styling**: Tailwind CSS with shadcn/ui component library
- **Build Tool**: Vite with custom plugins for Replit integration
- **Path Aliases**: `@/` maps to `client/src/`, `@shared/` maps to `shared/`

### Backend Architecture
- **Runtime**: Node.js with Express
- **Language**: TypeScript compiled with tsx
- **API Pattern**: RESTful endpoints defined in `shared/routes.ts`
- **Process Spawning**: Python simulations run as child processes with real-time output streaming
- **Build**: Custom esbuild script bundles server with selective dependency bundling for cold start optimization

### Data Storage
- **Database**: PostgreSQL via Drizzle ORM
- **Schema Location**: `shared/schema.ts`
- **Migrations**: Managed via `drizzle-kit push`
- **Session Storage**: connect-pg-simple for Express sessions

### Simulation Engine
- **Language**: Python 3
- **Core Library**: REBOUND N-body integrator
- **Configurations Tested**: λ-spacing (√φ), various MMR resonances (4:3, 3:2, 2:1), random spacing
- **Output**: JSON results and matplotlib visualizations saved to public directory

### Shared Code Pattern
- Schema definitions and route contracts live in `shared/` directory
- Both frontend and backend import from `@shared/` for type safety
- Zod schemas provide runtime validation

## External Dependencies

### AI Provider Integrations
- **OpenAI**: GPT models via Replit AI Integrations
- **Anthropic**: Claude models via SDK
- **Google**: Gemini models via `@google/genai` SDK
- **OpenRouter**: Access to DeepSeek, Qwen, Grok, Llama, Mistral models

### Scientific Computing
- **REBOUND**: N-body orbital integrator for Python
- **NumPy/SciPy**: Statistical analysis and numerical computing
- **Matplotlib**: Visualization generation

### Database
- **PostgreSQL**: Primary data store (requires DATABASE_URL environment variable)
- **Drizzle ORM**: Type-safe database queries
- **pg**: Node.js PostgreSQL driver

### Environment Variables Required
- `DATABASE_URL`: PostgreSQL connection string
- `AI_INTEGRATIONS_OPENAI_API_KEY` / `AI_INTEGRATIONS_OPENAI_BASE_URL`
- `AI_INTEGRATIONS_ANTHROPIC_API_KEY` / `AI_INTEGRATIONS_ANTHROPIC_BASE_URL`
- `AI_INTEGRATIONS_GEMINI_API_KEY` / `AI_INTEGRATIONS_GEMINI_BASE_URL`
- `AI_INTEGRATIONS_OPENROUTER_API_KEY` / `AI_INTEGRATIONS_OPENROUTER_BASE_URL`