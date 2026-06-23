const chatPanel = document.querySelector("#chatPanel");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const conversationList = document.querySelector("#conversationList");
const newChatButton = document.querySelector("#newChatButton");
const openHistoryButton = document.querySelector("#openHistoryButton");
const closeHistoryButton = document.querySelector("#closeHistoryButton");
const drawerMask = document.querySelector("#drawerMask");
const historyDrawer = document.querySelector("#historyDrawer");
const conversationSearch = document.querySelector("#conversationSearch");
const currentConversationTitle = document.querySelector("#currentConversationTitle");
const fileInput = document.querySelector("#fileInput");
const attachmentRow = document.querySelector("#attachmentRow");
const attachmentType = document.querySelector("#attachmentType");
const attachmentName = document.querySelector("#attachmentName");
const removeAttachmentButton = document.querySelector("#removeAttachmentButton");
const STORAGE_VERSION = "sqlite-v1";
const savedStorageVersion = localStorage.getItem("storage_version");
if (savedStorageVersion !== STORAGE_VERSION) {
  localStorage.removeItem("conversation_id");
  localStorage.setItem("storage_version", STORAGE_VERSION);
}
let conversationId = localStorage.getItem("conversation_id") || "default";
let conversationCache = [];
let pastedAttachment = null;

function isMobileLayout() {
  return window.matchMedia("(max-width: 860px)").matches;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  fileInput.disabled = isBusy;
  sendButton.textContent = isBusy ? "生成中" : "发送";
}

function setNewChatBusy(isBusy) {
  newChatButton.disabled = isBusy;
}

function scrollToBottom() {
  chatPanel.scrollTop = chatPanel.scrollHeight;
}

function clearMessages() {
  chatPanel.innerHTML = "";
}

function openHistory() {
  if (!isMobileLayout()) {
    return;
  }
  historyDrawer.classList.add("open");
  drawerMask.classList.add("open");
  conversationSearch.focus();
}

function closeHistory() {
  historyDrawer.classList.remove("open");
  drawerMask.classList.remove("open");
  messageInput.focus();
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeAssistantText(text) {
  return String(text || "暂无文本结果")
    .replace(/[✅🔹📌🎯💡✨🚀🔍🔥📞📎🌐⚡💫🔒]/g, "")
    .replace(/^\s*#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*\|[-\s|:]+\|\s*$/gm, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderMarkdown(text) {
  const escapedText = escapeHtml(normalizeAssistantText(text));
  return escapedText
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^---$/gm, "<hr>")
    .replace(/\n/g, "<br>");
}

function formatElapsedTime(milliseconds) {
  if (milliseconds < 1000) {
    return `${milliseconds} ms`;
  }
  return `${(milliseconds / 1000).toFixed(1)} 秒`;
}

function addMessage(role, text, imageUrl = "", meta = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.innerHTML = renderMarkdown(text);
  bubble.appendChild(paragraph);

  if (imageUrl) {
    const image = document.createElement("img");
    image.className = "result-image";
    image.src = imageUrl;
    image.alt = "生成图片";
    bubble.appendChild(image);
  }

  if (meta) {
    const metaNode = document.createElement("div");
    metaNode.className = "meta";
    metaNode.textContent = meta;
    bubble.appendChild(metaNode);
  }

  article.appendChild(bubble);
  chatPanel.appendChild(article);
  scrollToBottom();
  return article;
}

function addLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant loading-message";

  const bubble = document.createElement("div");
  bubble.className = "bubble loading-bubble";
  bubble.innerHTML = `
    <span class="loading-dot"></span>
    <span class="loading-dot"></span>
    <span class="loading-dot"></span>
    <span class="loading-text">生成中，耗时：0 ms</span>
  `;

  const startedAt = performance.now();
  const loadingText = bubble.querySelector(".loading-text");
  article._timer = window.setInterval(() => {
    const elapsed = Math.floor(performance.now() - startedAt);
    loadingText.textContent = `生成中，耗时：${formatElapsedTime(elapsed)}`;
  }, 500);

  article.appendChild(bubble);
  chatPanel.appendChild(article);
  scrollToBottom();
  return article;
}

function removeMessage(messageNode) {
  if (messageNode && messageNode._timer) {
    window.clearInterval(messageNode._timer);
  }
  if (messageNode && messageNode.parentNode) {
    messageNode.parentNode.removeChild(messageNode);
  }
}

async function sendMessage(message) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
  }

  return response.json();
}

async function sendPosterMessage(message, file) {
  const formData = new FormData();
  formData.append("message", message);
  formData.append("conversation_id", conversationId);
  formData.append("poster_type", "商业海报");
  formData.append("campaign", message);
  if (file) {
    formData.append("file", file);
  }

  const response = await fetch("/api/poster/generate", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function readErrorMessage(response) {
  const fallback = `请求失败：${response.status}`;
  const errorText = await response.text();
  if (!errorText) {
    return fallback;
  }

  try {
    const errorData = JSON.parse(errorText);
    return errorData.detail || errorData.message || errorText;
  } catch (error) {
    return errorText;
  }
}

function formatPosterText(data) {
  const copywriting = data.copywriting || {};
  const lines = [
    copywriting.headline ? `主标题：${copywriting.headline}` : "",
    copywriting.subheadline ? `副标题：${copywriting.subheadline}` : "",
    copywriting.cta ? `行动语：${copywriting.cta}` : "",
  ].filter(Boolean);
  return lines.join("\n") || "已生成海报方案。";
}

function clearAttachment() {
  fileInput.value = "";
  pastedAttachment = null;
  attachmentType.textContent = "文件";
  attachmentName.textContent = "";
  attachmentRow.hidden = true;
}

function getAttachmentType(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".pdf")) {
    return "PDF";
  }
  if (name.endsWith(".docx")) {
    return "DOCX";
  }
  if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")) {
    return "图片";
  }
  return "文件";
}

function updateAttachmentView() {
  const file = fileInput.files[0];
  if (!file) {
    clearAttachment();
    return;
  }
  pastedAttachment = null;
  attachmentType.textContent = getAttachmentType(file);
  attachmentName.textContent = file.name;
  attachmentRow.hidden = false;
}

function setPastedAttachment(file) {
  fileInput.value = "";
  pastedAttachment = file;
  attachmentType.textContent = "粘贴图片";
  attachmentName.textContent = file.name;
  attachmentRow.hidden = false;
}

async function fetchConversations() {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    throw new Error(`会话列表加载失败：${response.status}`);
  }
  return response.json();
}

async function fetchConversation(conversationId) {
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`);
  if (!response.ok) {
    throw new Error(`会话加载失败：${response.status}`);
  }
  return response.json();
}

async function createConversation() {
  const response = await fetch("/api/conversations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title: "新会话" }),
  });
  if (!response.ok) {
    throw new Error(`新建会话失败：${response.status}`);
  }
  return response.json();
}

async function deleteConversation(targetConversationId) {
  const response = await fetch(`/api/conversations/${encodeURIComponent(targetConversationId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`删除会话失败：${response.status}`);
  }
  return response.json();
}

function renderEmptyState() {
  clearMessages();
  const emptyState = document.createElement("div");
  emptyState.className = "empty-state";
  emptyState.innerHTML = `
    <div>
      <h2>开始新的电商创作</h2>
      <p>可以生成商品文案、商业海报，也可以继续优化上一版方案。</p>
    </div>
  `;
  chatPanel.appendChild(emptyState);
}

function renderConversationList(conversations) {
  conversationList.innerHTML = "";
  const keyword = conversationSearch.value.trim().toLowerCase();
  const visibleConversations = conversations.filter((conversation) => {
    const title = String(conversation.title || "").toLowerCase();
    return !keyword || title.includes(keyword);
  });

  visibleConversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item";
    if (conversation.conversation_id === conversationId) {
      button.classList.add("active");
    }

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = conversation.title || "新会话";

    const meta = document.createElement("span");
    meta.className = "conversation-meta";
    meta.textContent = `${conversation.message_count || 0} 条消息`;

    button.appendChild(title);
    button.appendChild(meta);

    const deleteButton = document.createElement("span");
    deleteButton.className = "conversation-delete";
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await removeConversation(conversation.conversation_id);
    });
    button.appendChild(deleteButton);

    button.addEventListener("click", async () => {
      await switchConversation(conversation.conversation_id);
      if (isMobileLayout()) {
        closeHistory();
      }
    });
    conversationList.appendChild(button);
  });
}

async function removeConversation(targetConversationId) {
  if (!targetConversationId) {
    return;
  }

  const shouldDelete = window.confirm("确定删除这个会话吗？删除后无法恢复。");
  if (!shouldDelete) {
    return;
  }

  await deleteConversation(targetConversationId);
  const conversations = await fetchConversations();
  conversationCache = conversations;

  if (targetConversationId === conversationId) {
    const nextConversation = conversations[0];
    conversationId = nextConversation ? nextConversation.conversation_id : "default";
    localStorage.setItem("conversation_id", conversationId);
    await switchConversation(conversationId);
    return;
  }

  renderConversationList(conversationCache);
}

function renderConversationMessages(messages) {
  clearMessages();
  if (!messages.length) {
    renderEmptyState();
    return;
  }
  messages.forEach((message) => {
    if (message.role === "user" || message.role === "assistant") {
      addMessage(message.role, message.content, message.image_url || "");
    }
  });
}

async function loadConversationList() {
  const conversations = await fetchConversations();
  conversationCache = conversations;
  renderConversationList(conversationCache);
}

async function switchConversation(nextConversationId) {
  conversationId = nextConversationId || "default";
  localStorage.setItem("conversation_id", conversationId);
  const conversation = await fetchConversation(conversationId);
  currentConversationTitle.textContent = conversation.title || "新会话";
  renderConversationMessages(conversation.messages || []);
  await loadConversationList();
}

async function initializeApp() {
  try {
    await loadConversationList();
    await switchConversation(conversationId);
  } catch (error) {
    clearMessages();
    addMessage("assistant", error.message || "会话加载失败，请稍后重试。");
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  const selectedFile = pastedAttachment || fileInput.files[0];
  if (!message && !selectedFile) {
    return;
  }

  const userText = selectedFile ? `${message || "请分析上传文件并生成海报"}\n附件：${selectedFile.name}` : message;
  addMessage("user", userText);
  messageInput.value = "";
  setBusy(true);
  const loadingMessage = addLoadingMessage();

  try {
    const data = selectedFile
      ? await sendPosterMessage(message || "请分析上传文件并生成海报", selectedFile)
      : await sendMessage(message);
    removeMessage(loadingMessage);
    if (selectedFile) {
      conversationId = data.conversation_id || conversationId;
      localStorage.setItem("conversation_id", conversationId);
      const imageUrl = data.poster ? data.poster.image_url : "";
      const generationTime = data.metadata ? data.metadata.generation_time : "-";
      addMessage("assistant", formatPosterText(data), imageUrl, `海报生成 ｜ 耗时：${generationTime}`);
      clearAttachment();
    } else {
      conversationId = data.conversation_id || conversationId;
      localStorage.setItem("conversation_id", conversationId);
      const meta = `耗时：${data.latency_ms || 0} ms`;
      addMessage("assistant", data.text, data.image_url, meta);
    }
    await loadConversationList();
  } catch (error) {
    removeMessage(loadingMessage);
    addMessage("assistant", error.message || "请求失败，请稍后重试。");
  } finally {
    setBusy(false);
    messageInput.focus();
  }
});

fileInput.addEventListener("change", updateAttachmentView);
removeAttachmentButton.addEventListener("click", clearAttachment);
messageInput.addEventListener("paste", (event) => {
  const items = event.clipboardData ? Array.from(event.clipboardData.items) : [];
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  if (!imageItem) {
    return;
  }

  const file = imageItem.getAsFile();
  if (!file) {
    return;
  }

  const extension = file.type === "image/png" ? "png" : "jpg";
  const pastedFile = new File(
    [file],
    `pasted-reference-${Date.now()}.${extension}`,
    { type: file.type || "image/png" }
  );
  setPastedAttachment(pastedFile);
});

newChatButton.addEventListener("click", async () => {
  setNewChatBusy(true);
  try {
    const conversation = await createConversation();
    await switchConversation(conversation.conversation_id);
    if (isMobileLayout()) {
      closeHistory();
    }
  } catch (error) {
    addMessage("assistant", error.message || "新建会话失败，请稍后重试。");
  } finally {
    setNewChatBusy(false);
    messageInput.disabled = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
});

openHistoryButton.addEventListener("click", openHistory);
closeHistoryButton.addEventListener("click", closeHistory);
drawerMask.addEventListener("click", closeHistory);

window.addEventListener("resize", () => {
  if (!isMobileLayout()) {
    historyDrawer.classList.remove("open");
    drawerMask.classList.remove("open");
  }
});

conversationSearch.addEventListener("input", () => {
  renderConversationList(conversationCache);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

initializeApp();
