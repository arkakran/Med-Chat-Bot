let isLoading = false;

// DOM elements
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const loadingOverlay = document.getElementById('loadingOverlay');
const characterCount = document.querySelector('.character-count');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateCharacterCount();
    setTimeout(checkHealth, 2000); // Check health after 2 seconds
});

function initializeEventListeners() {
    messageInput.addEventListener('keypress', handleKeyPress);
    messageInput.addEventListener('input', updateCharacterCount);
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function updateCharacterCount() {
    const count = messageInput.value.length;
    characterCount.textContent = `${count}/500`;
    
    if (count > 450) {
        characterCount.style.color = '#e74c3c';
    } else if (count > 400) {
        characterCount.style.color = '#f39c12';
    } else {
        characterCount.style.color = '#666';
    }
}

function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    if (message.length > 500) {
        showNotification('Message too long. Please keep it under 500 characters.', 'error');
        return;
    }
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    updateCharacterCount();
    
    // Show loading
    showLoading();
    
    // Send to backend
    fetchChatResponse(message);
}

function fetchChatResponse(message) {
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            addMessage(`I apologize, but I encountered an error: ${data.error}`, 'bot');
        } else {
            addMessage(data.response, 'bot', data.timestamp);
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Error:', error);
        addMessage('I apologize, but I encountered a connection error. Please check your internet connection and try again.', 'bot');
    });
}

function addMessage(text, sender, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-user-md"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    
    // Format text with proper HTML
    const formattedText = formatMessageText(text);
    textDiv.innerHTML = formattedText;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = timestamp || getCurrentTime();
    
    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

function formatMessageText(text) {
    // Convert line breaks to HTML
    let formatted = text.replace(/\n/g, '<br>');
    
    // Convert numbered lists
    formatted = formatted.replace(/^\d+\.\s+(.+?)$/gm, '<li>$1</li>');
    
    // Convert bullet points
    formatted = formatted.replace(/^[-•]\s+(.+?)$/gm, '<li>$1</li>');
    
    // Wrap consecutive list items in ul tags
    formatted = formatted.replace(/(<li>.*<\/li>)(\s*<br>\s*<li>.*<\/li>)*/gs, '<ul>$&</ul>');
    formatted = formatted.replace(/<br>\s*(?=<li>)/g, '');
    formatted = formatted.replace(/(?<=<\/li>)\s*<br>/g, '');
    
    // Handle medical disclaimer
    if (formatted.includes('---')) {
        formatted = formatted.replace(/---\s*\*(.*?)\*/g, '<div class="medical-disclaimer">$1</div>');
    }
    
    return formatted;
}

function getCurrentTime() {
    return new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function useSuggestion(suggestion) {
    messageInput.value = suggestion;
    updateCharacterCount();
    messageInput.focus();
}

function clearChat() {
    const messages = chatMessages.querySelectorAll('.message:not(:first-child)');
    messages.forEach((message, index) => {
        setTimeout(() => {
            message.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => message.remove(), 300);
        }, index * 50);
    });
}

function showLoading() {
    isLoading = true;
    sendButton.disabled = true;
    loadingOverlay.style.display = 'flex';
    addTypingIndicator();
}

function hideLoading() {
    isLoading = false;
    sendButton.disabled = false;
    loadingOverlay.style.display = 'none';
    removeTypingIndicator();
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-user-md"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <p style="font-style: italic; color: #666; margin-top: 8px;">Searching medical knowledge base...</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function checkHealth() {
    fetch('/health')
    .then(response => response.json())
    .then(data => {
        const status = data.status === 'healthy' ? 'System Healthy' : 'System Error';
        const chunks = data.vector_database?.total_chunks || 0;
        
        showNotification(
            `${status} - Knowledge Base: ${chunks} medical chunks loaded`,
            data.status === 'healthy' ? 'success' : 'error'
        );
    })
    .catch(error => {
        showNotification('Failed to check system health', 'error');
    });
}

function reprocessPDF() {
    if (confirm('This will reprocess the medical PDF and recreate the knowledge base. Continue?')) {
        showNotification('Reprocessing medical knowledge base...', 'info');
        showLoading();
        
        fetch('/reprocess_pdf', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showNotification(`Error: ${data.error}`, 'error');
            } else {
                showNotification(`Success! Reprocessed ${data.chunks_count} medical chunks`, 'success');
                addMessage(`✅ Medical knowledge base has been refreshed! I now have access to ${data.chunks_count} updated chunks of medical information.`, 'bot');
            }
        })
        .catch(error => {
            hideLoading();
            showNotification('Failed to reprocess PDF', 'error');
        });
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${getNotificationIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '10px',
        color: 'white',
        fontWeight: '600',
        zIndex: '1001',
        animation: 'slideInRight 0.4s ease',
        maxWidth: '350px',
        wordWrap: 'break-word',
        boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
    });
    
    // Set background color based on type
    const colors = {
        success: '#2ecc71',
        error: '#e74c3c',
        warning: '#f39c12',
        info: '#3498db'
    };
    notification.style.background = `linear-gradient(135deg, ${colors[type]}, ${darkenColor(colors[type] || colors.info, 20)})`;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease';
        setTimeout(() => notification.remove(), 400);
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return icons[type] || icons.info;
}

function darkenColor(color, percent) {
    // Simple color darkening function
    const num = parseInt(color.replace("#", ""), 16);
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
        (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
        (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
}

// Add CSS animations for notifications and effects
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes slideOut {
        from { transform: translateY(0); opacity: 1; }
        to { transform: translateY(-20px); opacity: 0; }
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        align-items: center;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3498db;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: scale(1); opacity: 0.5; }
        30% { transform: scale(1.2); opacity: 1; }
    }
`;
document.head.appendChild(style);
