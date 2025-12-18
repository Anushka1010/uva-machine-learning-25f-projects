import axios from 'axios';
import PQueue from 'p-queue';

// Interface for Canvas Course object
// Based on Canvas API documentation
export interface Course {
  id: number;
  name: string;
  account_id: number;
  uuid: string;
  start_at: string | null;
  grading_standard_id: number | null;
  is_public: boolean;
  created_at: string;
  course_code: string;
  default_view: string;
  root_account_id: number;
  enrollment_term_id: number;
  license: string;
  grade_passback_setting: string | null;
  end_at: string | null;
  public_syllabus: boolean;
  public_syllabus_to_auth: boolean;
  storage_quota_mb: number;
  is_public_to_auth_users: boolean;
  homeroom_course: boolean;
  course_color: string | null;
  friendly_name: string | null;
  apply_assignment_group_weights: boolean;
  calendar: {
    ics: string;
  };
  time_zone: string;
  blueprint: boolean;
  template: boolean;
  enrollments?: any[];
  hide_final_grades: boolean;
  workflow_state: string;
  restrict_enrollments_to_course_dates: boolean;
  overridden_course_visibility: string | null;
}

export interface Assignment {
  id: number;
  description: string;
  due_at: string | null;
  unlock_at: string | null;
  lock_at: string | null;
  points_possible: number;
  grading_type: string;
  assignment_group_id: number;
  grading_standard_id: number | null;
  created_at: string;
  updated_at: string;
  peer_reviews: boolean;
  automatic_peer_reviews: boolean;
  position: number;
  grade_group_students_individually: boolean;
  anonymous_peer_reviews: boolean;
  group_category_id: number | null;
  post_to_sis: boolean;
  moderated_grading: boolean;
  omit_from_final_grade: boolean;
  intra_group_peer_reviews: boolean;
  anonymous_instructor_annotations: boolean;
  anonymous_grading_delegation: boolean;
  graders_anonymous_to_graders: boolean;
  grader_count: number;
  grader_comments_visible_to_graders: boolean;
  final_grader_id: number | null;
  grader_names_visible_to_final_grader: boolean;
  allowed_attempts: number;
  annotatable_attachment_id: number | null;
  hide_in_gradebook: boolean;
  secure_params: string;
  course_id: number;
  name: string;
  submission_types: string[];
  has_submitted_submissions: boolean;
  dueDateRequired: boolean;
  max_name_length: number;
  in_closed_grading_period: boolean;
  is_quiz_assignment: boolean;
  can_duplicate: boolean;
  original_course_id: number | null;
  original_assignment_id: number | null;
  original_lti_resource_link_id: number | null;
  original_assignment_name: string | null;
  original_quiz_id: number | null;
  workflow_state: string;
  important_dates: boolean;
  muted: boolean;
  html_url: string;
  published: boolean;
  only_visible_to_overrides: boolean;
  visible_to_everyone: boolean;
  locked_for_user: boolean;
  submissions_download_url: string;
}

export interface Announcement {
  id: number;
  title: string;
  message: string;
  posted_at: string;
  author: {
    id: number;
    display_name: string;
  };
  context_code: string;
  delayed_post_at: string | null;
  read_state: string;
  locked: boolean;
  html_url: string;
}

export interface DiscussionTopic {
  id: number;
  title: string;
  message: string | null;
  posted_at: string;
  author: {
    id: number;
    display_name: string;
  };
  discussion_type: string;
  published: boolean;
  locked: boolean;
  html_url: string;
  user_name: string;
}

export interface Page {
  page_id: number;
  url: string;
  title: string;
  created_at: string;
  updated_at: string;
  published: boolean;
  body: string | null;
  front_page: boolean;
  html_url: string;
  editing_roles: string;
}

export interface Module {
  id: number;
  name: string;
  position: number;
  unlock_at: string | null;
  require_sequential_progress: boolean;
  published: boolean;
  items_count: number;
  items_url: string;
  items: ModuleItem[];
}

export interface ModuleItem {
  id: number;
  module_id: number;
  position: number;
  title: string;
  indent: number;
  type: string;
  content_id: number;
  html_url: string;
  url: string;
  published: boolean;
}

export interface Folder {
  id: number;
  name: string;
  full_name: string;
  context_id: number;
  context_type: string;
  parent_folder_id: number | null;
  created_at: string;
  updated_at: string;
  lock_at: string | null;
  unlock_at: string | null;
  position: number;
  locked: boolean;
  folders_url: string;
  files_url: string;
  files_count: number;
  folders_count: number;
  hidden: boolean;
  locked_for_user: boolean;
  hidden_for_user: boolean;
  for_submissions: boolean;
  can_upload: boolean;
}

export interface CanvasFile {
  id: number;
  uuid: string;
  folder_id: number;
  display_name: string;
  filename: string;
  'content-type': string;
  url: string;
  size: number;
  created_at: string;
  updated_at: string;
  unlock_at: string | null;
  locked: boolean;
  hidden: boolean;
  lock_at: string | null;
  hidden_for_user: boolean;
  thumbnail_url: string | null;
  modified_at: string;
  mime_class: string;
  media_entry_id: string | null;
}

export interface Quiz {
  id: number;
  title: string;
  html_url: string;
  mobile_url: string;
  description: string | null;
  quiz_type: string;
  time_limit: number | null;
  shuffle_answers: boolean;
  show_correct_answers: boolean;
  scoring_policy: string;
  allowed_attempts: number;
  one_question_at_a_time: boolean;
  question_count: number;
  points_possible: number;
  cant_go_back: boolean;
  access_code: string | null;
  ip_filter: string | null;
  due_at: string | null;
  lock_at: string | null;
  unlock_at: string | null;
  published: boolean;
  unpublishable: boolean;
  locked_for_user: boolean;
  lock_info: any;
  lock_explanation: string | null;
  hide_results: string | null;
  show_correct_answers_at: string | null;
  hide_correct_answers_at: string | null;
  all_dates: any[];
  can_unpublish: boolean;
  can_update: boolean;
  require_lockdown_browser: boolean;
  require_lockdown_browser_for_results: boolean;
  require_lockdown_browser_monitor: boolean;
  lockdown_browser_monitor_data: string | null;
}

// Initialize p-queue with concurrency of 5
const queue = new PQueue({ concurrency: 5 });

/* Helper to extract the 'next' page URL from the Link header. */
const getNextPageUrl = (linkHeader: string): string | null => {
  if (!linkHeader) return null;
  const links = linkHeader.split(',');
  const nextLink = links.find((link) => link.includes('rel="next"'));
  if (!nextLink) return null;
  const match = nextLink.match(/<([^>]+)>/);
  return match ? match[1] : null;
};

/* Helper to check if an error is a 403 or 404 access error. */
const isAccessError = (error: any): boolean => {
    return error.response && (error.response.status === 403 || error.response.status === 404);
};

/* Generic function to fetch data from Canvas API using a queue to prevent rate limiting.
   @param endpoint - The full URL to the endpoint.
   @param token - The Canvas API access token.
   @returns The response data. */
export const getGenericData = async <T>(endpoint: string, token: string): Promise<T> => {
  return queue.add(async () => {
    const response = await axios.get<T>(endpoint, {
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  }) as Promise<T>; // p-queue returns the result of the task
};

/* Fetches all assignments for a specific course, handling pagination.
   @param canvasDomain - The domain of the Canvas instance.
   @param token - The Canvas API access token.
   @param courseId - The ID of the course.
   @returns An array of Assignment objects. */
export const fetchAssignments = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Assignment[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  let url: string | null = `${baseUrl}/api/v1/courses/${courseId}/assignments?per_page=50`;
  const allAssignments: Assignment[] = [];

  while (url) {
    const currentUrl = url; // Capture for closure if needed, though we use await
    try {
        const response = await queue.add(async () => {
          return axios.get<Assignment[]>(currentUrl, {
            headers: {
              Authorization: `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          });
        });
    
        if (response) {
            allAssignments.push(...response.data);
            url = getNextPageUrl(response.headers.link as string);
        } else {
            url = null;
        }
    } catch (error: any) {
        if (isAccessError(error)) {
            console.warn(`[Sync] Access denied or not found for assignments in course ${courseId} (Status: ${error.response?.status})`);
            url = null; // Stop pagination
        } else {
            console.error(`Failed to fetch assignments for course ${courseId}:`, error);
            url = null;
        }
    }
  }

  return allAssignments;
};

/* Fetches active courses for the user.
   @param canvasDomain - The domain of the Canvas instance (e.g., "https://canvas.instructure.com").
   @param token - The Canvas API access token.
   @returns An array of active Course objects. */
export const fetchActiveCourses = async (canvasDomain: string, token: string): Promise<Course[]> => {
  // Ensure the domain has the protocol and no trailing slash
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/courses?enrollment_state=active`;
  return getGenericData<Course[]>(endpoint, token);
};

/* Fetches announcements for a course.
   @param canvasDomain - The domain of the Canvas instance.
   @param token - The Canvas API access token.
   @param courseId - The ID of the course.
   @returns An array of Announcement objects. */
export const fetchAnnouncements = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Announcement[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/announcements?context_codes[]=course_${courseId}&per_page=50`;
  
  console.log(`Fetching announcements for course: ${courseId}`);
  
  try {
    const data = await getGenericData<Announcement[]>(endpoint, token);
    console.log(`Found ${data.length} announcements`);
    return data;
  } catch (error: any) {
    if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for announcements in course ${courseId} (Status: ${error.response?.status})`);
        return [];
    }
    console.error(`Failed to fetch announcements for course ${courseId}:`, error);
    return [];
  }
};

export const fetchDiscussionTopics = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<DiscussionTopic[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/courses/${courseId}/discussion_topics?per_page=50`;
  
  console.log(`Fetching discussion topics for course: ${courseId}`);
  
  try {
    const data = await getGenericData<DiscussionTopic[]>(endpoint, token);
    console.log(`Found ${data.length} discussion topics`);
    return data;
  } catch (error: any) {
    if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for discussion topics in course ${courseId} (Status: ${error.response?.status})`);
        return [];
    }
    console.error(`Failed to fetch discussion topics for course ${courseId}:`, error);
    return [];
  }
};

export const fetchPages = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Page[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/courses/${courseId}/pages?per_page=50`;
  
  console.log(`Fetching pages for course: ${courseId}`);
  
  try {
    const data = await getGenericData<Page[]>(endpoint, token);
    console.log(`Found ${data.length} pages`);
    return data;
  } catch (error: any) {
    if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for pages in course ${courseId} (Status: ${error.response?.status})`);
        return [];
    }
    console.error(`Failed to fetch pages for course ${courseId}:`, error);
    return [];
  }
};

export const fetchModules = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Module[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  let url: string | null = `${baseUrl}/api/v1/courses/${courseId}/modules?per_page=50&include[]=items`;
  const allModules: Module[] = [];
  
  console.log(`Fetching modules for course: ${courseId}`);
  
  while (url) {
    const currentUrl = url;
    try {
      const response = await queue.add(async () => {
        return axios.get<Module[]>(currentUrl, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      });

      if (response) {
        allModules.push(...response.data);
        url = getNextPageUrl(response.headers.link as string);
      } else {
        url = null;
      }
    } catch (error: any) {
      if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for modules in course ${courseId} (Status: ${error.response?.status})`);
        url = null;
      } else {
        console.error(`Failed to fetch page of modules for course ${courseId}:`, error);
        url = null;
      }
    }
  }

  console.log(`Found ${allModules.length} modules`);
  return allModules;
};

export const fetchFile = async (
  canvasDomain: string,
  token: string,
  courseId: string,
  fileId: string
): Promise<CanvasFile | null> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/courses/${courseId}/files/${fileId}`;

  try {
    return await getGenericData<CanvasFile>(endpoint, token);
  } catch (error: any) {
    if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for file ${fileId} in course ${courseId} (Status: ${error.response?.status})`);
        return null;
    }
    console.error(`Failed to fetch file ${fileId} for course ${courseId}:`, error);
    return null;
  }
};

export const fetchFolders = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Folder[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  let url: string | null = `${baseUrl}/api/v1/courses/${courseId}/folders?per_page=50`;
  const allFolders: Folder[] = [];

  console.log(`Fetching folders for course: ${courseId}`);

  while (url) {
    const currentUrl = url;
    try {
      const response = await queue.add(async () => {
        return axios.get<Folder[]>(currentUrl, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      });

      if (response) {
        allFolders.push(...response.data);
        url = getNextPageUrl(response.headers.link as string);
      } else {
        url = null;
      }
    } catch (error: any) {
        if (isAccessError(error)) {
            console.warn(`[Sync] Access denied or not found for folders in course ${courseId} (Status: ${error.response?.status})`);
            url = null;
        } else {
            console.error(`Failed to fetch page of folders for course ${courseId}:`, error);
            url = null; 
        }
    }
  }
  
  console.log(`Found ${allFolders.length} folders`);
  return allFolders;
};

export const fetchFiles = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<CanvasFile[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  // Using the course files endpoint returns all files in the course,
  // which is more reliable than iterating folders for the root folder.
  let url: string | null = `${baseUrl}/api/v1/courses/${courseId}/files?per_page=50`;
  const allFiles: CanvasFile[] = [];
  
  console.log(`Fetching files for course: ${courseId}`);
  
  while (url) {
    const currentUrl = url;
    try {
      const response = await queue.add(async () => {
        return axios.get<CanvasFile[]>(currentUrl, {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      });

      if (response) {
        allFiles.push(...response.data);
        url = getNextPageUrl(response.headers.link as string);
      } else {
        url = null;
      }
    } catch (error: any) {
        if (isAccessError(error)) {
            console.warn(`[Sync] Access denied or not found for files in course ${courseId} (Status: ${error.response?.status})`);
            url = null;
        } else {
            console.error(`Failed to fetch page of files for course ${courseId}:`, error);
            url = null;
        }
    }
  }

  console.log(`Found ${allFiles.length} files`);
  return allFiles;
};

export const fetchQuizzes = async (
  canvasDomain: string,
  token: string,
  courseId: string
): Promise<Quiz[]> => {
  let baseUrl = canvasDomain;
  if (!baseUrl.startsWith('http')) {
    baseUrl = `https://${baseUrl}`;
  }
  baseUrl = baseUrl.replace(/\/$/, '');

  const endpoint = `${baseUrl}/api/v1/courses/${courseId}/quizzes?per_page=50`;
  
  console.log(`Fetching quizzes for course: ${courseId}`);
  
  try {
    const data = await getGenericData<Quiz[]>(endpoint, token);
    console.log(`Found ${data.length} quizzes`);
    return data;
  } catch (error: any) {
    if (isAccessError(error)) {
        console.warn(`[Sync] Access denied or not found for quizzes in course ${courseId} (Status: ${error.response?.status})`);
        return [];
    }
    console.error(`Failed to fetch quizzes for course ${courseId}:`, error);
    return [];
  }
};