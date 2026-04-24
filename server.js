//npm install express cors pg dotenv crypto
import express from 'express';
import cors from 'cors';
import pkg from 'pg';
import dotenv from 'dotenv';
import crypto from 'crypto';

dotenv.config();

const { Pool } = pkg;
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Database setup
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false }
});

async function initDB() {
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                authType TEXT NOT NULL
            );
        `);
        console.log("Database initialized");
    } catch (err) {
        console.error("Error initializing database:", err);
    }
}
initDB();

function hashPassword(password) {
    return crypto.createHash('sha256').update(password).digest('hex');
}

// Signup endpoint
app.post('/api/signup', async (req, res) => {
    const { username, email, password } = req.body;

    try {
        const existing = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        if (existing.rows.length > 0) {
            return res.status(400).json({ success: false, message: 'Email already in use' });
        }

        const hashedPassword = hashPassword(password);
        const result = await pool.query(
            'INSERT INTO users (username, email, password, authType) VALUES ($1, $2, $3, $4) RETURNING id',
            [username, email, hashedPassword, 'email']
        );

        res.json({ success: true, userId: result.rows[0].id });
    } catch (err) {
        res.status(500).json({ success: false, message: 'Server error' });
    }
});

// Email/password login endpoint
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;

    try {
        const hashedPassword = hashPassword(password);
        const result = await pool.query(
            'SELECT id, username FROM users WHERE email = $1 AND password = $2',
            [email, hashedPassword]
        );

        if (result.rows.length > 0) {
            res.json({success: true, user: result.rows[0] });
        } else {
            return res.status(400).json({ success: false, message: 'Invalid credentials' });
        }
    } catch (err) {
        res.status(500).json({ success: false, message: 'Server error' });
    }
});

//Google login endpoint
app.post('/api/google-login', async (req, res) => {
    const { username, email, authType } = req.body;

    try {
        const existing = await pool.query('SELECT * FROM users WHERE email = $1', [email]);

        if (existing.rows.length === 0) {
            await pool.query(
                'INSERT INTO users (username, email, password, authType) VALUES ($1, $2, $3, $4)',
                [username, email, 'google-auth', authType]
            );
        }

        res.json({ success: true});
    } catch (err) {
        res.status(500).json({ success: false, message: 'Server error' });
    }
});


app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

