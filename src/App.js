import React, { useState } from 'react';

function App() {
    const [name, setName] = useState('');
    const [message, setMessage] = useState('');

    const handleSubmit = async () => {
        const response = await fetch('http://localhost:5000/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name }),
        });
        const data = await response.json();
        setMessage(data.message);
    };

    return (
        <div>
            <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your name"
            />
            <button onClick={handleSubmit}>Submit</button>
            <p>{message}</p>
        </div>
    );
}

export default App;