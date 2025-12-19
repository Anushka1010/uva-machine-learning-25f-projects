import { Annotation, END, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { getKeys } from "../services/auth";
import type { IngestionRuleInput } from "./db_utils";

// --- Types ---

type ComponentType = "syllabus" | "schedule" | "assignments" | "course_home";

type ComponentStrategy = {
  isExternal: boolean;
  url?: string;
  reasoning?: string;
};

type AnalysisOutput = Record<ComponentType, ComponentStrategy>;

// --- State Definition ---

const DiscoveryState = Annotation.Root({
  courseId: Annotation<string>({
    reducer: (_x, y) => y ?? "",
    default: () => "",
  }),
  fullContext: Annotation<string>({
    reducer: (x, y) => x + "\n\n" + y,
    default: () => "",
  }),
  componentStrategies: Annotation<AnalysisOutput>({
    reducer: (x, y) => ({ ...x, ...y }),
    default: () => ({
      syllabus: { isExternal: false },
      schedule: { isExternal: false },
      assignments: { isExternal: false },
      course_home: { isExternal: false },
    }),
  }),
  finalRules: Annotation<IngestionRuleInput[]>({
    reducer: (_x, y) => y ?? [],
    default: () => [],
  }),
});

// --- Helpers ---

const verifyUrl = async (url: string): Promise<{ valid: boolean; contentType?: string }> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout
    
    // Try HEAD first to be fast
    let response = await fetch(url, {
      method: "HEAD",
      signal: controller.signal,
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" },
    });

    // Fallback to GET (some servers block HEAD)
    if (!response.ok && response.status !== 404) {
      response = await fetch(url, {
        method: "GET",
        headers: { "Range": "bytes=0-500" }, // Fetch just the header/start
        signal: controller.signal,
      });
    }
    
    clearTimeout(timeoutId);
    return { valid: response.ok, contentType: response.headers.get("content-type") || undefined };
  } catch (error) {
    return { valid: false };
  }
};

// --- Nodes ---

// 1. Fetch Data via Canvas API
async function fetchCourseData(state: typeof DiscoveryState.State) {
  console.log(`[Discovery] API Fan-out for course: ${state.courseId}`);
  const { canvasToken, canvasDomain } = getKeys();
  const courseId = state.courseId;
  
  if (!canvasDomain || !courseId) return { fullContext: "" };

  const headers = canvasToken ? { "Authorization": `Bearer ${canvasToken}` } : {};
  const baseUrl = `https://${canvasDomain}/api/v1/courses/${courseId}`;

  const requests = [
    { name: "SYLLABUS_AND_META", url: `${baseUrl}?include[]=syllabus_body` },
    { name: "HOME_PAGE", url: `${baseUrl}/pages/front_page` },
    { name: "ASSIGNMENT_LIST", url: `${baseUrl}/assignments?per_page=10` },
    // We explicitly check modules too, as links often live there
    { name: "MODULES_SAMPLE", url: `${baseUrl}/modules?per_page=5` } 
  ];

  const results = await Promise.all(requests.map(async (req) => {
    try {
      const res = await fetch(req.url, { headers: headers as Record<string, string> });
      if (!res.ok) return `--- ${req.name} ---\n[Empty/Failed]`;
      
      const json = await res.json();
      let content = "";

      if (req.name === "SYLLABUS_AND_META") content = json.syllabus_body || "No syllabus body.";
      else if (req.name === "HOME_PAGE") content = json.body || "No home body.";
      else if (req.name === "ASSIGNMENT_LIST") {
         content = Array.isArray(json) && json.length > 0 
          ? `Found ${json.length} items. Sample: ` + json.slice(0, 3).map((a:any) => `${a.name} (${a.html_url})`).join(", ")
          : "List empty.";
      }
      else if (req.name === "MODULES_SAMPLE") {
          content = Array.isArray(json) && json.length > 0
          ? `Found modules. Items: ` + JSON.stringify(json.slice(0,3)) // Rough dump to catch links
          : "List empty.";
      }

      return `--- ${req.name} ---\n${content.slice(0, 5000)}\n`;
    } catch (e) {
      return `--- ${req.name} ---\n[Error]`;
    }
  }));

  return { fullContext: results.join("\n\n") };
}

// 2. Analyze Consolidated Context
async function analyzeStructure(state: typeof DiscoveryState.State) {
  console.log("[Discovery] Analyzing structure for granular links");
  const context = state.fullContext;
  const keys = getKeys();

  // Defaults
  const defaults: AnalysisOutput = {
    syllabus: { isExternal: false },
    schedule: { isExternal: false },
    assignments: { isExternal: false },
    course_home: { isExternal: false }
  };

  if (!keys.openaiKey) return { componentStrategies: defaults };

  const llm = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0,
    apiKey: keys.openaiKey,
    modelKwargs: { response_format: { type: "json_object" } }
  });

  const prompt = `You are a crawler configuration agent. 
  
  OBJECTIVE: Find the SPECIFIC URLs for course components.
  If a professor uses an external site, they often have different pages for "Schedule", "Assignments", and "Home".
  You must extract the *specific* deep link for each if present.

  DATA CONTEXT:
  ${context.slice(0, 15000)}

  TASKS:
  1. syllabus: Link to PDF or 'syllabus.html' page?
  2. schedule: Link to a 'schedule.html', 'calendar', or 'lectures' page?
  3. assignments: Link to 'homework.html', 'labs.html', or external tool (Gradescope)?
  4. course_home: The main external landing page.

  Return JSON:
  {
    "syllabus": { "isExternal": boolean, "url": "string | null" },
    "schedule": { "isExternal": boolean, "url": "string | null" },
    "assignments": { "isExternal": boolean, "url": "string | null" },
    "course_home": { "isExternal": boolean, "url": "string | null" }
  }
  `;

  try {
    const response = await llm.invoke([{ role: "user", content: prompt }]);
    const contentText = typeof response.content === 'string' ? response.content : "";
    const jsonMatch = contentText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return { componentStrategies: defaults };

    return { componentStrategies: JSON.parse(jsonMatch[0]) };
  } catch (error) {
    console.error("[Discovery] Analysis failed", error);
    return { componentStrategies: defaults };
  }
}

// 3. Verify & Build Rules (The Granular Logic)
async function verifySource(state: typeof DiscoveryState.State) {
  console.log("[Discovery] Building exhaustive ingestion rules");
  const strategies = state.componentStrategies;
  const courseId = state.courseId;
  const finalRules: IngestionRuleInput[] = [];

  // --- A. ALWAYS: Native Canvas Routes (The Safety Net) ---
  // We explicitly save these routes so the scraper knows exactly what to check.
  
  finalRules.push({
    source_url: `canvas://api/courses/${courseId}/modules`,
    source_type: "CANVAS_API",
    category: "modules",
    extraction_config: { scope: "modules", note: "Primary structure" }
  });

  finalRules.push({
    source_url: `canvas://api/courses/${courseId}/files`,
    source_type: "CANVAS_API",
    category: "files",
    extraction_config: { scope: "files", note: "Orphaned files" }
  });
  
  finalRules.push({
      source_url: `canvas://api/courses/${courseId}/discussion_topics?only_announcements=true`,
      source_type: "CANVAS_API",
      category: "announcements",
      extraction_config: { scope: "announcements", note: "Announcements" }
  });

  // Only add Native Assignments if we didn't find an external one, 
  // OR add it anyway as a backup (safest option).
  finalRules.push({
    source_url: `canvas://api/courses/${courseId}/assignments`,
    source_type: "CANVAS_API",
    category: "assignments",
    extraction_config: { scope: "assignments", note: "Native assignments" }
  });

  // --- B. EXTERNAL ROUTES (Specific Pages) ---
  const components: ComponentType[] = ["syllabus", "schedule", "assignments", "course_home"];

  for (const comp of components) {
    const strategy = strategies[comp];
    
    if (strategy.isExternal && strategy.url) {
      let rawUrl = strategy.url.trim().replace(/['"]/g, "");
      
      // Cleanup & Validation
      if (rawUrl.startsWith("/")) continue; 
      if (!rawUrl.startsWith("http") || rawUrl.includes("canvas.instructure")) continue;

      const ver = await verifyUrl(rawUrl);
      
      if (ver.valid) {
        let sourceType: IngestionRuleInput["source_type"] = "EXTERNAL_HTML";
        let category: IngestionRuleInput["category"] = comp;
        if (rawUrl.endsWith(".pdf") || ver.contentType?.includes("pdf")) sourceType = "PDF_LINK";
        else if (rawUrl.includes("drive.google.com")) {
          sourceType = "GOOGLE_DRIVE";
          category = comp === "schedule" ? "lecture slides" : comp;
        }

        // KEY CHANGE: We allow multiple external rules if the URLs are different.
        // e.g., prof.com/index.html AND prof.com/schedule.html are both saved.
        const isDuplicate = finalRules.some(r => r.source_url === rawUrl);

        if (!isDuplicate) {
           console.log(`[Discovery] Adding specific rule for ${comp}: ${rawUrl}`);
           finalRules.push({
            source_url: rawUrl,
            source_type: sourceType,
            category,
            extraction_config: { 
              component_tag: comp, // Tag enables specific parsing logic downstream
              detected_content_type: ver.contentType 
            }
          });
        }
      }
    }
  }

  return { finalRules };
}

// --- Graph Definition ---

const graph = new StateGraph(DiscoveryState)
  .addNode("fetch_data", fetchCourseData)
  .addNode("analyze_structure", analyzeStructure)
  .addNode("verify_source", verifySource)
  .setEntryPoint("fetch_data")
  .addEdge("fetch_data", "analyze_structure")
  .addEdge("analyze_structure", "verify_source")
  .addEdge("verify_source", END);

export const discoveryAgent = graph.compile();
export type DiscoveryStateType = typeof DiscoveryState.State;
