import { Client } from '@notionhq/client';
import { getKeys } from './auth';

/*
 Initializes the Notion workspace by creating the necessary databases.
 @param pageId - The ID of the parent page where databases will be created.
 @returns An object containing the IDs of the created databases.
 */
export const initializeWorkspace = async (pageId: string) => {
  const { notionToken } = getKeys();
  
  if (!notionToken) {
    throw new Error('Notion token not found. Please save your keys first.');
  }

  const notion = new Client({ auth: notionToken });

  // 1. Create 'Courses' Database
  const coursesDb = await notion.databases.create({
    parent: { page_id: pageId },
    title: [
      {
        type: 'text',
        text: {
          content: 'Courses',
        },
      },
    ],
    properties: {
      Name: {
        title: {},
      },
      'Canvas ID': {
        rich_text: {},
      },
      'Course Code': {
        rich_text: {},
      },
    },
  });

  // 2. Create 'Resources' Database
  const resourcesDb = await notion.databases.create({
    parent: { page_id: pageId },
    title: [
      {
        type: 'text',
        text: {
          content: 'Resources',
        },
      },
    ],
    properties: {
      Name: {
        title: {},
      },
      'Canvas ID': {
        rich_text: {},
      },
      Type: {
        select: {
          options: [
            { name: 'File', color: 'blue' },
            { name: 'Link', color: 'yellow' },
          ],
        },
      },
      URL: {
        url: {},
      },
    },
  });

  // 3. Create 'Assignments' Database with Relation to Courses
  const assignmentsDb = await notion.databases.create({
    parent: { page_id: pageId },
    title: [
      {
        type: 'text',
        text: {
          content: 'Assignments',
        },
      },
    ],
    properties: {
      Name: {
        title: {},
      },
      'Canvas ID': {
        rich_text: {},
      },
      'Due Date': {
        date: {},
      },
      Status: {
        select: {
          options: [
            { name: 'Not Started', color: 'red' },
            { name: 'In Progress', color: 'yellow' },
            { name: 'Done', color: 'green' },
          ],
        },
      },
      Course: {
        relation: {
          database_id: coursesDb.id,
          // Use single_property to only show the relation on the Assignments database side initially
          // to avoid cluttering the Courses database if not explicitly requested.
          type: 'single_property',
          single_property: {} 
        },
      },
    },
  });

  return {
    coursesDbId: coursesDb.id,
    assignmentsDbId: assignmentsDb.id,
    resourcesDbId: resourcesDb.id,
  };
};
