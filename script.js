// Plant database with educational information
const plantDatabase = {
    'Aloevera': {
        name: 'Aloe Vera',
        description: 'Aloe vera is a succulent plant species known for its medicinal properties. The gel from its leaves is used for treating burns, wounds, and skin conditions.',
        properties: ['Anti-inflammatory', 'Antimicrobial', 'Wound healing', 'Skin care'],
        uses: 'Topical application for burns, cuts, and skin irritations. Also used in cosmetics and health supplements.',
        scientificName: 'Aloe barbadensis miller'
    },
    'Amla': {
        name: 'Indian Gooseberry',
        description: 'Amla is a highly nutritious fruit rich in vitamin C and antioxidants. It has been used in Ayurvedic medicine for centuries.',
        properties: ['Antioxidant', 'Immunity booster', 'Digestive aid', 'Hair care'],
        uses: 'Consumed fresh, as juice, or in powdered form. Used for boosting immunity and improving hair health.',
        scientificName: 'Phyllanthus emblica'
    },
    'Bamboo': {
        name: 'Bamboo',
        description: 'Bamboo is a fast-growing plant with various medicinal properties. Its shoots and leaves are used in traditional medicine.',
        properties: ['Anti-inflammatory', 'Antioxidant', 'Digestive aid', 'Detoxifying'],
        uses: 'Bamboo shoots are consumed as food, while leaves are used in traditional medicine for various ailments.',
        scientificName: 'Bambusoideae'
    },
    'Bhrami': {
        name: 'Brahmi',
        description: 'Brahmi is a renowned herb in Ayurveda known for its cognitive-enhancing properties and ability to improve memory and concentration.',
        properties: ['Memory enhancer', 'Anti-anxiety', 'Neuroprotective', 'Adaptogenic'],
        uses: 'Used to improve memory, reduce anxiety, and enhance cognitive function. Often consumed as tea or supplements.',
        scientificName: 'Bacopa monnieri'
    },
    'Bringaraja': {
        name: 'Eclipta',
        description: 'Bringaraja is a traditional herb used in Ayurveda for hair care and liver health. It is known for its hepatoprotective properties.',
        properties: ['Hair growth promoter', 'Hepatoprotective', 'Anti-inflammatory', 'Antioxidant'],
        uses: 'Used for promoting hair growth, treating liver disorders, and improving overall health.',
        scientificName: 'Eclipta prostrata'
    },
    'Castor': {
        name: 'Castor Plant',
        description: 'The castor plant produces castor oil, which has been used for centuries for its medicinal and industrial properties.',
        properties: ['Laxative', 'Anti-inflammatory', 'Moisturizing', 'Antimicrobial'],
        uses: 'Castor oil is used as a laxative, skin moisturizer, and in various cosmetic products.',
        scientificName: 'Ricinus communis'
    },
    'Coffee': {
        name: 'Coffee Plant',
        description: 'Coffee plants produce beans that are rich in caffeine and antioxidants. Coffee has various health benefits when consumed in moderation.',
        properties: ['Stimulant', 'Antioxidant', 'Metabolic booster', 'Cognitive enhancer'],
        uses: 'Coffee beans are roasted and brewed to make coffee, which is consumed for its stimulating effects and health benefits.',
        scientificName: 'Coffea arabica'
    },
    'Coriender': {
        name: 'Coriander',
        description: 'Coriander is a versatile herb used both as a spice and medicine. Both its leaves and seeds have medicinal properties.',
        properties: ['Digestive aid', 'Anti-inflammatory', 'Antimicrobial', 'Antioxidant'],
        uses: 'Used in cooking as a spice and in traditional medicine for digestive issues and inflammation.',
        scientificName: 'Coriandrum sativum'
    },
    'Curry': {
        name: 'Curry Leaf',
        description: 'Curry leaves are aromatic leaves used in Indian cuisine and traditional medicine. They are rich in antioxidants and have various health benefits.',
        properties: ['Antioxidant', 'Anti-diabetic', 'Digestive aid', 'Hair care'],
        uses: 'Used in cooking for flavor and in traditional medicine for diabetes management and digestive health.',
        scientificName: 'Murraya koenigii'
    },
    'Eucalyptus': {
        name: 'Eucalyptus',
        description: 'Eucalyptus is known for its aromatic leaves and medicinal properties. It is commonly used for respiratory health.',
        properties: ['Decongestant', 'Antimicrobial', 'Anti-inflammatory', 'Expectorant'],
        uses: 'Used in aromatherapy, cough syrups, and topical applications for respiratory and skin conditions.',
        scientificName: 'Eucalyptus globulus'
    },
    'Ginger': {
        name: 'Ginger',
        description: 'Ginger is a popular spice and medicine known for its anti-inflammatory and digestive properties.',
        properties: ['Anti-inflammatory', 'Digestive aid', 'Anti-nausea', 'Antioxidant'],
        uses: 'Consumed fresh, dried, or as tea for digestive issues, nausea, and inflammation.',
        scientificName: 'Zingiber officinale'
    },
    'Guava': {
        name: 'Guava',
        description: 'Guava is a tropical fruit rich in vitamin C and antioxidants. Both the fruit and leaves have medicinal properties.',
        properties: ['Antioxidant', 'Anti-diabetic', 'Digestive aid', 'Immunity booster'],
        uses: 'Fruit is consumed fresh, while leaves are used in traditional medicine for diabetes and digestive issues.',
        scientificName: 'Psidium guajava'
    },
    'Henna': {
        name: 'Henna',
        description: 'Henna is a plant whose leaves are used for natural hair coloring and have cooling properties.',
        properties: ['Cooling', 'Antimicrobial', 'Hair conditioning', 'Anti-inflammatory'],
        uses: 'Used for natural hair coloring, skin cooling, and in traditional medicine for various skin conditions.',
        scientificName: 'Lawsonia inermis'
    },
    'Hibiscus': {
        name: 'Hibiscus',
        description: 'Hibiscus flowers are rich in antioxidants and vitamin C. They are used in teas and traditional medicine.',
        properties: ['Antioxidant', 'Anti-hypertensive', 'Diuretic', 'Hair care'],
        uses: 'Used to make tea, hair care products, and in traditional medicine for blood pressure management.',
        scientificName: 'Hibiscus rosa-sinensis'
    },
    'Lemon': {
        name: 'Lemon',
        description: 'Lemon is a citrus fruit rich in vitamin C and citric acid. It has various health benefits and medicinal uses.',
        properties: ['Antioxidant', 'Digestive aid', 'Detoxifying', 'Antimicrobial'],
        uses: 'Used in cooking, as juice, and in traditional medicine for digestive health and immunity.',
        scientificName: 'Citrus limon'
    },
    'Mint': {
        name: 'Mint',
        description: 'Mint is an aromatic herb known for its cooling properties and digestive benefits.',
        properties: ['Cooling', 'Digestive aid', 'Antimicrobial', 'Anti-inflammatory'],
        uses: 'Used in teas, cooking, and traditional medicine for digestive issues and respiratory health.',
        scientificName: 'Mentha spicata'
    },
    'Neem': {
        name: 'Neem',
        description: 'Neem is a powerful medicinal tree known for its antimicrobial and anti-inflammatory properties.',
        properties: ['Antimicrobial', 'Anti-inflammatory', 'Antioxidant', 'Hepatoprotective'],
        uses: 'Used in traditional medicine for skin conditions, dental care, and various infections.',
        scientificName: 'Azadirachta indica'
    },
    'Onion': {
        name: 'Onion',
        description: 'Onion is a common vegetable with various medicinal properties. It is rich in antioxidants and sulfur compounds.',
        properties: ['Antioxidant', 'Anti-inflammatory', 'Antimicrobial', 'Cardiovascular health'],
        uses: 'Used in cooking and traditional medicine for respiratory health and as an antimicrobial agent.',
        scientificName: 'Allium cepa'
    },
    'Palak(Spinach)': {
        name: 'Spinach',
        description: 'Spinach is a leafy green vegetable rich in iron, vitamins, and minerals. It has various health benefits.',
        properties: ['Iron-rich', 'Antioxidant', 'Anti-inflammatory', 'Bone health'],
        uses: 'Consumed as a vegetable and used in traditional medicine for anemia and bone health.',
        scientificName: 'Spinacia oleracea'
    },
    'Papaya': {
        name: 'Papaya',
        description: 'Papaya is a tropical fruit rich in enzymes, vitamins, and antioxidants. It has digestive and health benefits.',
        properties: ['Digestive aid', 'Antioxidant', 'Anti-inflammatory', 'Immunity booster'],
        uses: 'Consumed fresh and used in traditional medicine for digestive health and wound healing.',
        scientificName: 'Carica papaya'
    }
};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const leafImage = document.getElementById('leafImage');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const results = document.getElementById('results');
const loadingOverlay = document.getElementById('loadingOverlay');
const plantsGrid = document.getElementById('plantsGrid');
const plantSearch = document.getElementById('plantSearch');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeUploadArea();
    initializePlantDatabase();
    initializeSearch();
});

// Navigation functionality
function initializeNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    hamburger.addEventListener('click', function() {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function() {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });
}

// Upload area functionality
function initializeUploadArea() {
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        leafImage.click();
    });

    // File input change
    leafImage.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    results.style.display = 'none';
    leafImage.value = '';
}

// Classification functionality (placeholder for now)
function classifyLeaf() {
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Simulate API call (replace with actual model inference)
    setTimeout(() => {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
        
        // Show results (placeholder data)
        showResults('Aloevera', 0.95);
    }, 2000);
}

function showResults(plantName, confidence) {
    const plantInfo = plantDatabase[plantName];
    
    if (!plantInfo) {
        console.error('Plant not found in database:', plantName);
        return;
    }

    // Update plant name and description
    document.getElementById('plantName').textContent = plantInfo.name;
    document.getElementById('plantDescription').textContent = plantInfo.description;
    
    // Update confidence meter
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const confidencePercent = Math.round(confidence * 100);
    
    confidenceFill.style.width = confidencePercent + '%';
    confidenceText.textContent = confidencePercent + '%';
    
    // Generate top predictions (placeholder)
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    const topPredictions = [
        { name: plantInfo.name, confidence: confidence },
        { name: 'Mint', confidence: 0.15 },
        { name: 'Neem', confidence: 0.08 },
        { name: 'Ginger', confidence: 0.05 }
    ];
    
    topPredictions.forEach(prediction => {
        const predictionItem = document.createElement('div');
        predictionItem.className = 'prediction-item';
        predictionItem.innerHTML = `
            <span class="prediction-name">${prediction.name}</span>
            <span class="prediction-confidence">${Math.round(prediction.confidence * 100)}%</span>
        `;
        predictionsList.appendChild(predictionItem);
    });
    
    // Show results
    results.style.display = 'block';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth' });
}

// Plant database functionality
function initializePlantDatabase() {
    displayPlants(Object.keys(plantDatabase));
}

function displayPlants(plantKeys) {
    plantsGrid.innerHTML = '';
    
    plantKeys.forEach(plantKey => {
        const plant = plantDatabase[plantKey];
        const plantCard = createPlantCard(plant);
        plantsGrid.appendChild(plantCard);
    });
}

function createPlantCard(plant) {
    const card = document.createElement('div');
    card.className = 'plant-card';
    
    card.innerHTML = `
        <h4>${plant.name}</h4>
        <p><strong>Scientific Name:</strong> ${plant.scientificName}</p>
        <p>${plant.description}</p>
        <div class="plant-properties">
            ${plant.properties.map(prop => `<span class="property-tag">${prop}</span>`).join('')}
        </div>
    `;
    
    return card;
}

// Search functionality
function initializeSearch() {
    plantSearch.addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const filteredPlants = Object.keys(plantDatabase).filter(plantKey => {
            const plant = plantDatabase[plantKey];
            return plant.name.toLowerCase().includes(searchTerm) ||
                   plant.scientificName.toLowerCase().includes(searchTerm) ||
                   plant.description.toLowerCase().includes(searchTerm) ||
                   plant.properties.some(prop => prop.toLowerCase().includes(searchTerm));
        });
        
        displayPlants(filteredPlants);
    });
}

// Smooth scrolling for navigation links
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Add smooth scrolling to all navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        scrollToSection(targetId);
    });
});

// Add some interactive animations
function addFloatingAnimation() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 2}s`;
    });
}

// Initialize animations
document.addEventListener('DOMContentLoaded', function() {
    addFloatingAnimation();
});

// Add scroll animations
function addScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.plant-card, .feature, .stat').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Initialize scroll animations
document.addEventListener('DOMContentLoaded', function() {
    addScrollAnimations();
});
