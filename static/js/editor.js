function displayDailymotionResults(clips) {
    const resultsContainer = document.getElementById('dailymotion-results');
    resultsContainer.innerHTML = '';
    
    clips.forEach(clip => {
        const clipElement = document.createElement('div');
        clipElement.className = 'clip-item mb-3 p-3 border rounded';
        
        clipElement.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <img src="${clip.thumbnail}" alt="${clip.title}" class="img-fluid rounded">
                </div>
                <div class="col-md-9">
                    <h5>${clip.title}</h5>
                    <p>Duration: ${clip.duration}</p>
                    <div class="btn-group">
                        <a href="${clip.view_url}" target="_blank" class="btn btn-info btn-sm">
                            <i class="fas fa-eye"></i> View
                        </a>
                        <button onclick="downloadDailymotionClip('${clip.id}')" class="btn btn-primary btn-sm">
                            <i class="fas fa-download"></i> Download
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(clipElement);
    });
} 