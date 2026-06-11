document.addEventListener('DOMContentLoaded', () => {
    // ──────────────────────────────────────────────
    // DOM Elements
    // ──────────────────────────────────────────────
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const chatList = document.getElementById('chat-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const deleteChatBtn = document.getElementById('delete-chat-btn');
    const currentChatTitle = document.getElementById('current-chat-title');
    const messagesContainer = document.getElementById('messages-container');
    const welcomeScreen = document.getElementById('welcome-screen');
    const welcomeSubtitle = document.getElementById('welcome-subtitle');
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const inputWrapper = document.querySelector('.input-wrapper');
    const disclaimerText = document.getElementById('disclaimer-text');

    // Mode toggle elements
    const modeBtnRag = document.getElementById('mode-btn-rag');
    const modeBtnLlm = document.getElementById('mode-btn-llm');
    const modeSlider = document.getElementById('mode-slider');

    // RAG Upload elements (now in input area)
    const ragUploadArea = document.getElementById('rag-upload-area');
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadStatus = document.getElementById('upload-status');
    const docsSection = document.getElementById('docs-section');
    const docsList = document.getElementById('docs-list');

    // Nav elements
    const libraryBtn = document.getElementById('library-btn');
    const themeBtn = document.getElementById('theme-btn');

    // ──────────────────────────────────────────────
    // API Base URL
    // ──────────────────────────────────────────────
    const API_BASE = window.location.origin;

    // ──────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────
    let sessions = JSON.parse(localStorage.getItem('nexus_chat_sessions')) || [];
    let currentSessionId = null;
    let currentMode = localStorage.getItem('nexus_chat_mode') || 'rag'; // 'rag' or 'llm'
    let sidebarCollapsed = localStorage.getItem('nexus_sidebar_collapsed') === 'true';
    let currentTheme = localStorage.getItem('nexus_theme') || 'default';

    // ──────────────────────────────────────────────
    // Initialize
    // ──────────────────────────────────────────────
    init();

    function init() {
        // Apply saved sidebar state
        applySidebarState();

        // Apply saved mode
        applyMode(currentMode);

        // Apply saved theme
        applyTheme(currentTheme);

        renderSidebar();
        loadDocuments();

        // If there are sessions, load the most recent one, else leave on welcome screen
        if (sessions.length > 0) {
            switchSession(sessions[0].id);
        } else {
            showWelcomeScreen();
        }

        // Event Listeners
        sidebarToggle.addEventListener('click', toggleSidebar);
        newChatBtn.addEventListener('click', createNewSession);
        deleteChatBtn.addEventListener('click', deleteCurrentSession);
        chatForm.addEventListener('submit', handleSendMessage);

        // Mode toggle buttons
        modeBtnRag.addEventListener('click', () => switchMode('rag'));
        modeBtnLlm.addEventListener('click', () => switchMode('llm'));

        // Nav buttons
        if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
        if (libraryBtn) libraryBtn.addEventListener('click', () => {
            docsSection.style.display = docsSection.style.display === 'none' ? 'block' : 'none';
        });

        promptInput.addEventListener('input', () => {
            // Auto resize textarea
            promptInput.style.height = 'auto';
            promptInput.style.height = (promptInput.scrollHeight) + 'px';

            // Enable/disable send button
            sendBtn.disabled = promptInput.value.trim() === '';
        });

        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) {
                    handleSendMessage(e);
                }
            }
        });

        // PDF Upload
        pdfUpload.addEventListener('change', handleFileUpload);
    }

    // ──────────────────────────────────────────────
    // Sidebar Collapse/Expand
    // ──────────────────────────────────────────────

    function toggleSidebar() {
        sidebarCollapsed = !sidebarCollapsed;
        applySidebarState();
        localStorage.setItem('nexus_sidebar_collapsed', sidebarCollapsed);
    }

    function applySidebarState() {
        if (sidebarCollapsed) {
            sidebar.classList.add('collapsed');
            sidebarToggle.innerHTML = "<i class='bx bx-menu'></i>";
            sidebarToggle.title = "Open Sidebar";
        } else {
            sidebar.classList.remove('collapsed');
            sidebarToggle.innerHTML = "<i class='bx bx-chevron-left'></i>";
            sidebarToggle.title = "Close Sidebar";
        }
    }

    // ──────────────────────────────────────────────
    // Mode Toggle (RAG / LLM)
    // ──────────────────────────────────────────────

    function switchMode(mode) {
        if (mode === currentMode) return;
        currentMode = mode;
        localStorage.setItem('nexus_chat_mode', mode);
        applyMode(mode);
    }

    function applyMode(mode) {
        if (mode === 'llm') {
            // Activate LLM mode
            modeBtnRag.classList.remove('active');
            modeBtnLlm.classList.add('active');
            modeSlider.classList.add('llm');
            ragUploadArea.classList.add('hidden');
            document.body.classList.add('llm-mode');
            inputWrapper.classList.add('llm-mode');
            sendBtn.classList.add('llm-mode');
            promptInput.placeholder = 'Ask Nexus AI anything...';
            disclaimerText.textContent = 'Powered by a local LLM via Ollama. Responses are AI-generated.';
            welcomeSubtitle.textContent = 'Ask me anything — powered by your local LLM.';
        } else {
            // Activate RAG mode
            modeBtnLlm.classList.remove('active');
            modeBtnRag.classList.add('active');
            modeSlider.classList.remove('llm');
            ragUploadArea.classList.remove('hidden');
            document.body.classList.remove('llm-mode');
            inputWrapper.classList.remove('llm-mode');
            sendBtn.classList.remove('llm-mode');
            promptInput.placeholder = 'Message Nexus AI...';
            disclaimerText.textContent = 'Answers are extracted from your uploaded documents using NLP. No LLM is used.';
            welcomeSubtitle.textContent = 'Upload a PDF document and ask questions about its content.';
        }
    }

    // ──────────────────────────────────────────────
    // Theme Toggle
    // ──────────────────────────────────────────────

    function toggleTheme() {
        currentTheme = currentTheme === 'default' ? 'turquoise' : 'default';
        localStorage.setItem('nexus_theme', currentTheme);
        applyTheme(currentTheme);
    }

    function applyTheme(theme) {
        if (theme === 'turquoise') {
            document.documentElement.setAttribute('data-theme', 'turquoise');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
    }

    // ──────────────────────────────────────────────
    // Session Management
    // ──────────────────────────────────────────────

    function saveSessions() {
        localStorage.setItem('nexus_chat_sessions', JSON.stringify(sessions));
        renderSidebar();
    }

    function createNewSession() {
        const newSession = {
            id: Date.now().toString(),
            title: 'New Conversation',
            messages: []
        };
        sessions.unshift(newSession);
        saveSessions();
        switchSession(newSession.id);
        promptInput.focus();
    }

    function deleteCurrentSession() {
        if (!currentSessionId) return;

        if (confirm('Are you sure you want to delete this chat?')) {
            sessions = sessions.filter(s => s.id !== currentSessionId);
            currentSessionId = null;
            saveSessions();

            if (sessions.length > 0) {
                switchSession(sessions[0].id);
            } else {
                showWelcomeScreen();
            }
        }
    }

    function switchSession(id) {
        currentSessionId = id;
        const session = sessions.find(s => s.id === id);

        if (!session) return;

        // Update UI
        currentChatTitle.textContent = session.title;
        deleteChatBtn.style.display = 'block';
        welcomeScreen.classList.add('hidden');

        // Clear current messages (keeping welcome screen in DOM but hidden)
        const messages = messagesContainer.querySelectorAll('.message-wrapper');
        messages.forEach(m => m.remove());

        // Render session messages
        session.messages.forEach(msg => {
            if (msg.role === 'bot') {
                appendBotMessage(msg.content, msg.sources, msg.confidence_label, msg.mode || 'rag', false);
            } else {
                appendMessage('user', msg.content, false);
            }
        });

        renderSidebar();
        scrollToBottom();
    }

    function showWelcomeScreen() {
        currentSessionId = null;
        currentChatTitle.textContent = 'New Conversation';
        deleteChatBtn.style.display = 'none';
        welcomeScreen.classList.remove('hidden');

        const messages = messagesContainer.querySelectorAll('.message-wrapper');
        messages.forEach(m => m.remove());
        renderSidebar();
    }

    // ──────────────────────────────────────────────
    // Sidebar Rendering
    // ──────────────────────────────────────────────

    function renderSidebar() {
        chatList.innerHTML = '';

        sessions.forEach(session => {
            const li = document.createElement('li');
            li.className = `chat-item ${session.id === currentSessionId ? 'active' : ''}`;
            li.innerHTML = `
                <i class='bx bx-message-square-detail'></i>
                <span class="chat-title">${escapeHTML(session.title)}</span>
            `;
            li.addEventListener('click', () => switchSession(session.id));
            chatList.appendChild(li);
        });
    }

    // ──────────────────────────────────────────────
    // Chat Messaging (routes to RAG or LLM backend)
    // ──────────────────────────────────────────────

    async function handleSendMessage(e) {
        e.preventDefault();
        const content = promptInput.value.trim();
        if (!content) return;

        // Create a new session if none is active
        if (!currentSessionId) {
            const newSession = {
                id: Date.now().toString(),
                title: content.substring(0, 30) + (content.length > 30 ? '...' : ''),
                messages: []
            };
            sessions.unshift(newSession);
            currentSessionId = newSession.id;

            // Update UI for new session
            currentChatTitle.textContent = newSession.title;
            deleteChatBtn.style.display = 'block';
            welcomeScreen.classList.add('hidden');
        } else {
            // Update title if it's the first message
            const session = sessions.find(s => s.id === currentSessionId);
            if (session.messages.length === 0) {
                session.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
                currentChatTitle.textContent = session.title;
            }
        }

        const session = sessions.find(s => s.id === currentSessionId);

        // Reset input
        promptInput.value = '';
        promptInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Add user message
        const userMsg = { role: 'user', content };
        session.messages.push(userMsg);
        appendMessage('user', content, true);
        saveSessions();

        // Show typing indicator
        showTypingIndicator();

        try {
            let response;

            if (currentMode === 'llm') {
                // ─── LLM Mode ───
                // Build conversation history for multi-turn context
                const history = session.messages
                    .filter(m => m.role === 'user' || m.role === 'bot')
                    .slice(0, -1) // Exclude the current message (already in 'content')
                    .map(m => ({
                        role: m.role === 'bot' ? 'assistant' : 'user',
                        content: m.content
                    }));

                response = await fetch(`${API_BASE}/api/llm/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: content, history })
                });
            } else {
                // ─── RAG Mode ───
                response = await fetch(`${API_BASE}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: content })
                });
            }

            removeTypingIndicator();

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Store the full response data
            const botMsg = {
                role: 'bot',
                content: data.answer,
                sources: data.sources || [],
                confidence_label: data.confidence_label || 'none',
                mode: currentMode
            };

            // Only update UI if we're still on the same session
            if (currentSessionId === session.id) {
                appendBotMessage(data.answer, data.sources, data.confidence_label, currentMode, true);
            }

            session.messages.push(botMsg);
            saveSessions();

        } catch (error) {
            removeTypingIndicator();
            console.error('Chat error:', error);

            const errorMsg = {
                role: 'bot',
                content: `Sorry, something went wrong: ${error.message}`,
                sources: [],
                confidence_label: 'none',
                mode: currentMode
            };

            if (currentSessionId === session.id) {
                appendBotMessage(errorMsg.content, [], 'none', currentMode, true);
            }

            session.messages.push(errorMsg);
            saveSessions();
        }
    }

    // ──────────────────────────────────────────────
    // Message Rendering
    // ──────────────────────────────────────────────

    function appendMessage(role, content, animate) {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${role}`;
        if (!animate) wrapper.style.animation = 'none';

        const avatarIcon = role === 'bot' ? "<i class='bx bx-bot'></i>" : "<i class='bx bx-user'></i>";

        wrapper.innerHTML = `
            <div class="message-avatar">
                ${avatarIcon}
            </div>
            <div class="message-content">
                ${escapeHTML(content).replace(/\n/g, '<br>')}
            </div>
        `;

        messagesContainer.appendChild(wrapper);
        scrollToBottom();
    }

    function appendBotMessage(content, sources, confidenceLabel, mode, animate) {
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper bot';
        if (!animate) wrapper.style.animation = 'none';

        // Build the sources HTML (only show for RAG mode)
        let sourcesHTML = '';
        if (mode === 'rag' && sources && sources.length > 0) {
            const tags = sources.map(s =>
                `<span class="source-tag"><i class='bx bx-file'></i>${escapeHTML(s.file)} p.${s.page}</span>`
            ).join('');
            sourcesHTML = `<div class="message-sources">${tags}</div>`;
        }

        // Build confidence badge (only for RAG mode)
        let confidenceHTML = '';
        if (mode === 'rag' && confidenceLabel && confidenceLabel !== 'none') {
            const icons = { high: 'bx-check-circle', medium: 'bx-info-circle', low: 'bx-error-circle' };
            confidenceHTML = `
                <div class="confidence-badge ${confidenceLabel}">
                    <i class='bx ${icons[confidenceLabel] || 'bx-info-circle'}'></i>
                    ${confidenceLabel} confidence
                </div>
            `;
        }

        wrapper.innerHTML = `
            <div class="message-avatar">
                <i class='bx bx-bot'></i>
            </div>
            <div class="message-content">
                ${escapeHTML(content).replace(/\n/g, '<br>')}
                ${sourcesHTML}
                ${confidenceHTML}
            </div>
        `;

        messagesContainer.appendChild(wrapper);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper bot typing';
        wrapper.id = 'typing-indicator';

        wrapper.innerHTML = `
            <div class="message-avatar">
                <i class='bx bx-bot'></i>
            </div>
            <div class="message-content typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;

        messagesContainer.appendChild(wrapper);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // ──────────────────────────────────────────────
    // PDF Upload
    // ──────────────────────────────────────────────

    async function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showUploadStatus('Only PDF files are supported.', 'error');
            return;
        }

        showUploadStatus(`<span class="spinner"></span>Uploading & processing ${escapeHTML(file.name)}...`, 'uploading');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Upload failed');
            }

            showUploadStatus(`✓ ${escapeHTML(data.filename)} — ${data.chunks} chunks indexed.`, 'success');
            loadDocuments();

            // Auto-hide after 4 seconds
            setTimeout(() => hideUploadStatus(), 4000);

        } catch (error) {
            console.error('Upload error:', error);
            showUploadStatus(`✗ Upload failed: ${escapeHTML(error.message)}`, 'error');
        }

        // Reset the file input so the same file can be re-uploaded
        pdfUpload.value = '';
    }

    function showUploadStatus(message, type) {
        uploadStatus.innerHTML = message;
        uploadStatus.className = `upload-status ${type}`;
        uploadStatus.style.display = 'block';
    }

    function hideUploadStatus() {
        uploadStatus.style.display = 'none';
    }

    // ──────────────────────────────────────────────
    // Document Management
    // ──────────────────────────────────────────────

    async function loadDocuments() {
        try {
            const response = await fetch(`${API_BASE}/api/documents`);
            const data = await response.json();

            const docs = data.documents || [];

            if (docs.length > 0) {
                docsSection.style.display = 'block';
                docsList.innerHTML = '';

                docs.forEach(docName => {
                    const li = document.createElement('li');
                    li.className = 'doc-item';
                    li.innerHTML = `
                        <i class='bx bxs-file-pdf'></i>
                        <span class="doc-name">${escapeHTML(docName)}</span>
                        <button class="doc-delete-btn" title="Remove document">
                            <i class='bx bx-x'></i>
                        </button>
                    `;

                    const deleteBtn = li.querySelector('.doc-delete-btn');
                    deleteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteDocument(docName);
                    });

                    docsList.appendChild(li);
                });
            } else {
                docsList.innerHTML = '<li class="doc-item" style="justify-content: center; opacity: 0.5"><span class="doc-name">No documents yet</span></li>';
            }
        } catch (error) {
            // Backend might not be running yet — silently ignore
            console.log('Could not load documents:', error.message);
            docsList.innerHTML = '<li class="doc-item" style="justify-content: center; opacity: 0.5"><span class="doc-name">Failed to load</span></li>';
        }
    }

    async function deleteDocument(filename) {
        if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;

        try {
            const response = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                loadDocuments();
                showUploadStatus(`✓ ${escapeHTML(filename)} removed.`, 'success');
                setTimeout(() => hideUploadStatus(), 3000);
            } else {
                throw new Error('Failed to delete document');
            }
        } catch (error) {
            console.error('Delete error:', error);
            showUploadStatus(`✗ Failed to remove: ${escapeHTML(error.message)}`, 'error');
        }
    }

    // ──────────────────────────────────────────────
    // Utilities
    // ──────────────────────────────────────────────

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g,
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }
});
