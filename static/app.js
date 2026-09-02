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
const attachmentList = document.querySelector("#attachmentList");
const modelSwitcherButton = document.querySelector("#modelSwitcherButton");
const selectedModelLabel = document.querySelector("#selectedModelLabel");
const modelPicker = document.querySelector("#modelPicker");
const autoModelToggle = document.querySelector("#autoModelToggle");
const modelAutoLabel = document.querySelector("#modelAutoLabel");
const imageModelInputs = Array.from(document.querySelectorAll('input[name="imageModel"]'));
const openKnowledgeButton = document.querySelector("#openKnowledgeButton");
const closeKnowledgeButton = document.querySelector("#closeKnowledgeButton");
const knowledgeDialog = document.querySelector("#knowledgeDialog");
const knowledgeUploadForm = document.querySelector("#knowledgeUploadForm");
const knowledgeFileInput = document.querySelector("#knowledgeFileInput");
const knowledgeFileLabel = document.querySelector("#knowledgeFileLabel");
const knowledgeTitleInput = document.querySelector("#knowledgeTitleInput");
const knowledgeCategoryInput = document.querySelector("#knowledgeCategoryInput");
const knowledgeUploadButton = document.querySelector("#knowledgeUploadButton");
const knowledgeStatus = document.querySelector("#knowledgeStatus");
const knowledgeEntryMeta = document.querySelector("#knowledgeEntryMeta");
const knowledgeSourceCount = document.querySelector("#knowledgeSourceCount");
const knowledgeSourceList = document.querySelector("#knowledgeSourceList");
const STORAGE_VERSION = "mysql-redis-v2";
const savedStorageVersion = localStorage.getItem("storage_version");
if (savedStorageVersion !== STORAGE_VERSION) {
  localStorage.removeItem("conversation_id");
  localStorage.setItem("storage_version", STORAGE_VERSION);
}
let conversationId = localStorage.getItem("conversation_id") || "default";
let conversationCache = [];
let selectedAttachments = [];
const MAX_REFERENCE_ATTACHMENTS = 3;
const IMAGE_MODEL_LABELS = new Map([
  ["", "自动匹配"],
  ["qwen-image-3.0", "Qwen Image 3.0"],
  ["gpt-image-2", "GPT Image 2"],
]);
let selectedImageModel = localStorage.getItem("image_model") || "";
if (!IMAGE_MODEL_LABELS.has(selectedImageModel)) {
  selectedImageModel = "";
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 860px)").matches;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  fileInput.disabled = isBusy;
  modelSwitcherButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "生成中" : "发送";
}

function setNewChatBusy(isBusy) {
  newChatButton.disabled = isBusy;
}

function updateModelPicker() {
  imageModelInputs.forEach((input) => {
    input.checked = input.value === selectedImageModel;
  });
  autoModelToggle.checked = selectedImageModel === "";
  modelAutoLabel.textContent = selectedImageModel ? "手动指定" : "自动匹配";
  selectedModelLabel.textContent = IMAGE_MODEL_LABELS.get(selectedImageModel) || "自动匹配";
}

function closeModelPicker() {
  modelPicker.hidden = true;
  modelSwitcherButton.setAttribute("aria-expanded", "false");
}

function toggleModelPicker() {
  modelPicker.hidden = !modelPicker.hidden;
  modelSwitcherButton.setAttribute("aria-expanded", String(!modelPicker.hidden));
}

function setKnowledgeBusy(isBusy) {
  knowledgeFileInput.disabled = isBusy;
  knowledgeTitleInput.disabled = isBusy;
  knowledgeCategoryInput.disabled = isBusy;
  knowledgeUploadButton.disabled = isBusy;
  knowledgeUploadButton.textContent = isBusy ? "正在入库" : "上传入库";
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
      image_model: selectedImageModel,
    }),
  });

  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
  }

  return response.json();
}

async function sendPosterMessage(message, files) {
  const formData = new FormData();
  formData.append("message", message);
  formData.append("conversation_id", conversationId);
  formData.append("poster_type", "商业海报");
  formData.append("campaign", message);
  formData.append("image_model", selectedImageModel);
  files.forEach((file) => formData.append("files", file));

  const response = await fetch("/api/poster/generate", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  const task = await response.json();
  const taskId = task.task_id;
  if (!taskId) {
    throw new Error("服务器未返回图片生成任务编号。");
  }

  while (true) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    const taskResponse = await fetch(`/api/poster/tasks/${encodeURIComponent(taskId)}`);
    if (!taskResponse.ok) {
      throw new Error(await readErrorMessage(taskResponse));
    }
    const taskState = await taskResponse.json();
    if (taskState.status === "completed") {
      return taskState.result;
    }
    if (["failed", "cancelled"].includes(taskState.status)) {
      throw new Error(taskState.error || "图片生成任务失败。");
    }
  }
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
  return lines.join("\n") || data.metadata?.message || "已生成海报方案。";
}

async function openKnowledgeDialog() {
  if (isMobileLayout()) {
    closeHistory();
  }
  knowledgeDialog.hidden = false;
  document.body.classList.add("dialog-open");
  knowledgeStatus.textContent = "";
  knowledgeStatus.className = "knowledge-status";
  closeKnowledgeButton.focus();
  await loadKnowledgeSources();
}

function closeKnowledgeDialog() {
  knowledgeDialog.hidden = true;
  document.body.classList.remove("dialog-open");
  openKnowledgeButton.focus();
}

async function loadKnowledgeSources() {
  knowledgeSourceList.innerHTML = '<p class="knowledge-source-empty">正在加载</p>';
  try {
    const response = await fetch("/api/knowledge/sources");
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }
    renderKnowledgeSources(await response.json());
  } catch (error) {
    knowledgeSourceCount.textContent = "-";
    knowledgeEntryMeta.textContent = "暂不可用";
    knowledgeSourceList.innerHTML = "";
    const empty = document.createElement("p");
    empty.className = "knowledge-source-empty";
    empty.textContent = error.message || "资料列表加载失败";
    knowledgeSourceList.appendChild(empty);
  }
}

function renderKnowledgeSources(sources) {
  knowledgeSourceList.innerHTML = "";
  knowledgeSourceCount.textContent = `${sources.length} 项`;
  knowledgeEntryMeta.textContent = `${sources.length} 项资料`;
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "knowledge-source-empty";
    empty.textContent = "暂无资料";
    knowledgeSourceList.appendChild(empty);
    return;
  }

  sources.forEach((source) => {
    const item = document.createElement("div");
    item.className = "knowledge-source-item";

    const title = document.createElement("strong");
    title.textContent = source.title || source.source_filename || "未命名资料";
    const meta = document.createElement("span");
    meta.textContent = [source.category || "未分类", source.source_filename || ""].filter(Boolean).join(" · ");
    const count = document.createElement("span");
    count.className = "knowledge-vector-count";
    count.textContent = `${source.vector_count || 0} 条向量`;

    item.appendChild(title);
    item.appendChild(meta);
    item.appendChild(count);
    knowledgeSourceList.appendChild(item);
  });
}

async function uploadKnowledgeFile(file, title, category) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title);
  formData.append("category", category);
  const response = await fetch("/api/knowledge/upload", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return response.json();
}

function clearAttachment() {
  fileInput.value = "";
  selectedAttachments.forEach((item) => {
    if (item.previewUrl) {
      URL.revokeObjectURL(item.previewUrl);
    }
  });
  selectedAttachments = [];
  attachmentList.replaceChildren();
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

function isImageFile(file) {
  return file.type.startsWith("image/") || /\.(png|jpe?g)$/i.test(file.name);
}

function removeAttachment(index) {
  const [removed] = selectedAttachments.splice(index, 1);
  if (removed && removed.previewUrl) {
    URL.revokeObjectURL(removed.previewUrl);
  }
  renderAttachments();
}

function renderAttachments() {
  attachmentList.replaceChildren();
  selectedAttachments.forEach((item, index) => {
    const tile = document.createElement("div");
    const image = isImageFile(item.file);
    tile.className = image ? "attachment-tile image-tile" : "attachment-tile file-tile";
    tile.title = item.file.name;

    if (image) {
      const preview = document.createElement("img");
      preview.src = item.previewUrl;
      preview.alt = `本轮参考图片：${item.file.name}`;
      tile.appendChild(preview);
    } else {
      const type = document.createElement("span");
      type.className = "attachment-type";
      type.textContent = getAttachmentType(item.file);
      const name = document.createElement("span");
      name.className = "attachment-name";
      name.textContent = item.file.name;
      tile.append(type, name);
    }

    const remove = document.createElement("button");
    remove.className = "attachment-remove";
    remove.type = "button";
    remove.setAttribute("aria-label", `移除附件 ${item.file.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => removeAttachment(index));
    tile.appendChild(remove);
    attachmentList.appendChild(tile);
  });
  attachmentRow.hidden = selectedAttachments.length === 0;
}

function addAttachments(files) {
  const available = MAX_REFERENCE_ATTACHMENTS - selectedAttachments.length;
  if (available <= 0) {
    return;
  }
  files.slice(0, available).forEach((file) => {
    selectedAttachments.push({
      file,
      previewUrl: isImageFile(file) ? URL.createObjectURL(file) : "",
    });
  });
  renderAttachments();
}

function updateAttachmentView() {
  addAttachments(Array.from(fileInput.files));
  fileInput.value = "";
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
    await loadKnowledgeSources();
  } catch (error) {
    clearMessages();
    addMessage("assistant", error.message || "会话加载失败，请稍后重试。");
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  const selectedFiles = selectedAttachments.map((item) => item.file);
  if (!message && !selectedFiles.length) {
    return;
  }

  const userText = selectedFiles.length
    ? `${message || "请分析上传文件并生成海报"}\n附件：${selectedFiles.map((file) => file.name).join("、")}`
    : message;
  addMessage("user", userText);
  messageInput.value = "";
  setBusy(true);
  const loadingMessage = addLoadingMessage();

  try {
    const data = selectedFiles.length
      ? await sendPosterMessage(message || "请分析上传文件并生成海报", selectedFiles)
      : await sendMessage(message);
    removeMessage(loadingMessage);
    if (selectedFiles.length) {
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
modelSwitcherButton.addEventListener("click", toggleModelPicker);
imageModelInputs.forEach((input) => {
  input.addEventListener("change", () => {
    selectedImageModel = input.value;
    localStorage.setItem("image_model", selectedImageModel);
    updateModelPicker();
    closeModelPicker();
  });
});
autoModelToggle.addEventListener("change", () => {
  if (autoModelToggle.checked) {
    selectedImageModel = "";
    localStorage.setItem("image_model", selectedImageModel);
    updateModelPicker();
    return;
  }
  if (!selectedImageModel) {
    selectedImageModel = "qwen-image-3.0";
    localStorage.setItem("image_model", selectedImageModel);
    updateModelPicker();
  }
});
document.addEventListener("click", (event) => {
  if (!modelPicker.hidden && !modelPicker.contains(event.target) && !modelSwitcherButton.contains(event.target)) {
    closeModelPicker();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modelPicker.hidden) {
    closeModelPicker();
    modelSwitcherButton.focus();
  }
});
openKnowledgeButton.addEventListener("click", openKnowledgeDialog);
closeKnowledgeButton.addEventListener("click", closeKnowledgeDialog);
knowledgeFileInput.addEventListener("change", () => {
  const file = knowledgeFileInput.files[0];
  knowledgeFileLabel.textContent = file ? file.name : "选择文档或图片";
  if (file && !knowledgeTitleInput.value.trim()) {
    knowledgeTitleInput.value = file.name.replace(/\.[^.]+$/, "");
  }
});
knowledgeUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = knowledgeFileInput.files[0];
  if (!file) {
    knowledgeStatus.className = "knowledge-status error";
    knowledgeStatus.textContent = "请选择需要入库的资料。";
    return;
  }

  setKnowledgeBusy(true);
  knowledgeStatus.className = "knowledge-status";
  knowledgeStatus.textContent = "正在解析并向量化资料";
  try {
    const result = await uploadKnowledgeFile(
      file,
      knowledgeTitleInput.value.trim(),
      knowledgeCategoryInput.value.trim()
    );
    knowledgeStatus.className = "knowledge-status success";
    knowledgeStatus.textContent = `已入库：${result.text_vector_count} 条文本向量，${result.image_vector_count} 条图片向量。`;
    knowledgeUploadForm.reset();
    knowledgeFileLabel.textContent = "选择文档或图片";
    await loadKnowledgeSources();
  } catch (error) {
    knowledgeStatus.className = "knowledge-status error";
    knowledgeStatus.textContent = error.message || "资料入库失败";
  } finally {
    setKnowledgeBusy(false);
  }
});
knowledgeDialog.addEventListener("click", (event) => {
  if (event.target === knowledgeDialog) {
    closeKnowledgeDialog();
  }
});
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
  addAttachments([pastedFile]);
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

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !knowledgeDialog.hidden) {
    closeKnowledgeDialog();
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
updateModelPicker();
