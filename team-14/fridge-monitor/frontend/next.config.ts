/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Whitelist both just in case
    allowedOrigins: ["localhost:3000", "127.0.0.1:3000"],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // STRICTLY use 127.0.0.1 here. Do not use localhost.
        destination: 'http://127.0.0.1:5000/api/:path*',
      },
    ]
  },
};

export default nextConfig;
