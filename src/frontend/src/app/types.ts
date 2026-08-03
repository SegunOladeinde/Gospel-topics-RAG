export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}
