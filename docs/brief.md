# Project Brief: Stellar Connect

## Executive Summary

Stellar Connect is a sophisticated local RAG (Retrieval-Augmented Generation) system for trust and estate planning sales intelligence that transforms how sales operations manage and analyze client interactions. The platform processes 700+ existing sales call transcripts and 5-10 new daily transcripts to extract revenue-generating insights, automate client communications, and identify viral social media content opportunities. By leveraging local LLMs, vector databases, and intelligent agent orchestration, Stellar Connect eliminates 30+ minutes of manual work per call while building a comprehensive knowledge base that identifies patterns in successful closes and enables data-driven sales optimization.

## Problem Statement

### Current State and Pain Points
- Managing 3,900+ sales records in Notion with unstructured text fields that cannot be queried effectively
- Processing 5-10 new transcripts daily requiring 30+ minutes of manual email recaps and data entry per call
- Missing viral social media content opportunities buried in 700+ existing transcripts
- No ability to query across conversations to find patterns in successful closes or effective rebuttals
- Inability to track prospect readiness scores or conversation progression systematically
- Manual CRM updates taking hours per week with incomplete data capture
- Lost revenue from delayed follow-ups and missed engagement opportunities

### Impact Quantification
- **Time Loss:** 150-300 minutes daily on manual administrative tasks (5-10 calls × 30 min)
- **Content Opportunity Cost:** Missing 50+ potential viral posts weekly from unprocessed conversations
- **Close Rate Impact:** Unable to identify and replicate winning patterns across sales team
- **Revenue Loss:** Estimated 10-15% lower close rate due to delayed follow-ups and inconsistent messaging

### Why Existing Solutions Fall Short
- Cloud-based solutions violate client data privacy requirements for sensitive estate planning information
- Generic CRMs lack trust/estate-specific intelligence and pattern recognition
- Traditional transcription services provide text without actionable insights
- Current Notion setup cannot handle complex queries or pattern analysis
- No existing solution integrates transcript processing, CRM updates, content generation, and sales intelligence in one platform

### Urgency
With 5-10 new transcripts arriving daily and a backlog of 700+ unprocessed transcripts, the opportunity cost compounds daily. Each unprocessed transcript represents lost social content, delayed client engagement, and missed intelligence that could improve close rates.

## Proposed Solution

### Core Concept
Stellar Connect is a completely local, privacy-first intelligence system that uses CrewAI agent orchestration, PostgreSQL for granular data storage, Qdrant for semantic search, and Mistral-7B (MLX-optimized) for analysis. The system automatically processes transcripts as they arrive, generates send-ready email recaps, extracts viral content scored by platform suitability, and builds a queryable knowledge base accessible through a Gradio chat interface.

### Key Differentiators
- **100% Local Processing:** All data stays on the M2 Max MacBook, ensuring complete privacy
- **Trust Sales Specialization:** Custom-trained for estate planning terminology, objection patterns, and emotional triggers
- **Immediate Value Focus:** Email automation saves 30 minutes per call from day one
- **Evolutionary Architecture:** Starts with 2 CrewAI agents, scales to 7+ as value is proven
- **MLX Optimization:** 3-5x performance boost using Apple's Metal Performance Shaders
- **Existing Data Integration:** Preserves and enhances 3,900+ Notion records while adding granular structure

### Success Factors
- Processes transcripts in under 2 minutes with memory-efficient batching
- Generates email recaps requiring zero editing
- Scores content virality with trust/estate-specific patterns
- Enables natural language queries across entire knowledge base
- Tracks complete sales funnel including failed meetings

## Target Users

### Primary User Segment: Trust Sales Advisor (You)
- **Profile:** Estate planning sales professional managing high-volume sales operations
- **Current Workflow:** Manual transcript review, email writing, Notion data entry, content searching
- **Specific Needs:**
  - Automated administrative tasks to focus on selling
  - Instant access to successful rebuttals and patterns
  - Consistent, professional client communications
  - Social content that generates leads
- **Goals:** Close more deals, save time on admin work, build social media presence, identify winning patterns

### Secondary User Segment: Sales Team Members
- **Profile:** Other advisors who could benefit from shared intelligence
- **Current Behaviors:** Operating in silos without shared best practices
- **Needs:** Access to proven scripts, successful rebuttals, training materials
- **Goals:** Improve individual close rates through collective intelligence

## Goals & Success Metrics

### Business Objectives
- Reduce administrative time by 75% (from 30 to 7 minutes per call) within Week 1
- Generate 50+ viral social posts weekly from processed transcripts by Week 2
- Improve close rate by 10%+ through pattern analysis within 60 days
- Process 100% of daily transcripts within 2 minutes of receipt
- Build comprehensive knowledge base of 2,500+ transcripts within 1 year

### User Success Metrics
- Email recap generation requiring <1 minute of editing
- Sub-second response time for knowledge base queries
- 95% accuracy in parsing existing Notion data fields
- Zero data loss during migration from Notion to PostgreSQL
- Successful extraction of 5-10 social content pieces per transcript

### Key Performance Indicators (KPIs)
- **Processing Speed:** <2 minutes per transcript from file detection to completion
- **Email Quality:** 95% send-ready without edits
- **Content Virality:** 20% of extracted content achieving >100 engagement score
- **Query Response:** <1 second for semantic search across all transcripts
- **System Uptime:** 99.9% availability during business hours
- **Memory Efficiency:** <15GB active memory usage during peak processing

## MVP Scope

### Core Features (Must Have)
- **Automated Email Generation:** Process transcript → Generate personalized recap → Save 30 min/call
- **Viral Content Extraction:** Identify and score social media opportunities from transcripts
- **Notion Data Migration:** Import and structure 3,900+ existing sales records
- **PostgreSQL Database:** Granular tables for prospects, estates, properties, family, objections
- **Basic CrewAI Orchestration:** 2-3 agents (orchestrator, communication, content)
- **Transcript File Monitoring:** Watch folder for new .txt files and auto-process
- **Gradio Chat Interface:** Natural language queries against knowledge base

### Out of Scope for MVP
- Neo4j knowledge graph (deferred until foundation proven)
- Multi-user authentication and permissions
- Cloud synchronization or remote access
- Real-time transcription (working with existing .txt files)
- Automated social media posting (manual copy/paste for now)
- Complex sales coaching modules
- Video/audio processing (text transcripts only)
- Mobile application

### MVP Success Criteria
- Successfully process 50 test transcripts with accurate data extraction
- Generate 50 email recaps requiring minimal editing
- Extract 250+ pieces of scoreable social content
- Migrate all Notion records with 95% field accuracy
- Answer complex queries like "Show me all Ohio prospects over 60 with estates over $3M"

## Post-MVP Vision

### Phase 2 Features
- Expand to 7 CrewAI agents for specialized workflows
- Implement Qdrant vector search for semantic queries
- Add sentiment analysis and conversation phase detection
- Create automated follow-up sequences based on stage
- Build objection rebuttal library with effectiveness scores
- Generate weekly performance dashboards

### Long-term Vision (6-12 months)
- Neo4j knowledge graph for complex relationship mapping
- Predictive close probability scoring
- Automated coaching recommendations based on call analysis
- A/B testing framework for email and social content
- Integration with calendar for automated follow-up scheduling
- Revenue attribution from social content to closed deals

### Expansion Opportunities
- Multi-advisor team deployment with shared intelligence
- Industry expansion beyond trust/estate planning
- SaaS offering for other trust sales professionals (after proving locally)
- AI-powered role-play training simulations
- Competitive intelligence tracking from prospect mentions

## Technical Considerations

### Platform Requirements
- **Target Platforms:** macOS (M2 Max MacBook Pro primary)
- **Browser/OS Support:** Modern browsers for Gradio interface (Chrome, Safari, Firefox)
- **Performance Requirements:** Process transcript in <2 minutes, query response <1 second, handle 10 concurrent agent tasks

### Technology Preferences
- **Frontend:** Gradio for chat interface (simple, fast, local)
- **Backend:** Python with async/await, FastAPI for any APIs
- **Database:** PostgreSQL (primary), Qdrant (vectors), SQLite (agent state)
- **Hosting/Infrastructure:** 100% local on M2 Max, no cloud dependencies

### Architecture Considerations
- **Repository Structure:** Monorepo with clear separation of agents, database, RAG, and UI
- **Service Architecture:** Modular services that can run independently or orchestrated
- **Integration Requirements:** File system watching, PostgreSQL, Qdrant, CrewAI, MLX
- **Security/Compliance:** All data local, no external API calls, encrypted at rest

## Constraints & Assumptions

### Constraints
- **Budget:** $0 for cloud services (must be 100% local)
- **Timeline:** Week 1 must deliver email automation for immediate value
- **Resources:** Single developer/user, 32GB RAM limit
- **Technical:** Must handle Google Drive sync folder with escaped spaces, M2 Max optimization required

### Key Assumptions
- Transcripts will continue arriving as .txt files in consistent format
- File names will maintain "[Lead Name]: Estate Planning Advisor Meeting.txt" pattern
- 32GB RAM is sufficient for concurrent model + database operations
- MLX optimization will provide expected 3-5x performance boost
- Existing Notion data structure will remain stable during migration
- CrewAI framework will scale from 2 to 7+ agents without major refactoring

## Risks & Open Questions

### Key Risks
- **Memory Overflow:** Running multiple models simultaneously might exceed 32GB limit
- **Data Quality:** Unstructured Notion notes may have inconsistent patterns affecting parsing accuracy
- **Transcript Matching:** Duplicate or similar lead names could cause incorrect associations
- **Model Performance:** Mistral-7B might require fine-tuning for trust/estate terminology
- **Processing Backlog:** 700+ transcript backlog might take days to fully process

### Open Questions
- Optimal batch size for processing transcripts without memory issues?
- Best approach for handling transcript files with encoding issues?
- How to match transcripts to prospects when dates aren't in filenames?
- Should we implement incremental backups during migration?
- Optimal chunk size for transcript semantic search (1000 tokens vs 500)?

### Areas Needing Further Research
- MLX quantization settings for optimal memory/performance balance
- CrewAI agent communication patterns for complex workflows
- Prompt engineering for trust/estate-specific content extraction
- Viral content scoring algorithms for financial/legal niche
- PostgreSQL indexing strategies for 2,500+ transcripts/year

## Appendices

### A. Research Summary

**Market Research Findings:**
- Trust/estate planning industry processes millions of sales calls annually
- Average advisor spends 40% of time on administrative tasks
- Social media is becoming primary lead generation channel for financial services
- Privacy concerns are paramount when dealing with estate planning data

**Competitive Analysis:**
- No existing solution combines transcript processing, CRM, content generation, and sales intelligence
- Cloud-based alternatives (Gong, Chorus) are expensive and raise privacy concerns
- Generic CRMs (Salesforce, HubSpot) lack specialized trust/estate features

**Technical Feasibility:**
- MLX framework confirmed to provide 3-5x speedup on M2 Max
- CrewAI successfully tested with 7+ agent orchestration
- Mistral-7B performs well on trust/estate terminology with proper prompting
- PostgreSQL + Qdrant combination proven for similar RAG applications

### B. References
- CrewAI Documentation: https://docs.crewai.com
- MLX Framework: https://github.com/ml-explore/mlx
- Mistral-7B Model Card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- Qdrant Vector Database: https://qdrant.tech/documentation/
- Gradio Interface Guide: https://gradio.app/docs/

## Next Steps

### Immediate Actions
1. Set up PostgreSQL database with designed schema
2. Create migration script for Notion CSV data
3. Implement basic CrewAI system with 2 agents (orchestrator + communication)
4. Test email generation on 10 sample transcripts
5. Set up file watcher for transcript directory

### PM Handoff

This Project Brief provides the full context for Stellar Connect. Please start in 'PRD Generation Mode', review the brief thoroughly to work with the user to create the PRD section by section as the template indicates, asking for any necessary clarification or suggesting improvements.

The system combines existing Notion sales data (3,900+ records) with automated transcript processing to create a comprehensive trust sales intelligence platform. The focus is on immediate value through email automation (Week 1) while building toward a sophisticated multi-agent system that enables natural language queries and pattern analysis across all sales interactions.