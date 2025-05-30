module.exports = {
  port: process.env.PORT || 80,
  logLevel: 'silent',
  files: ['./dist/**/*.{html,htm,css,js}'],
  server: { 
    baseDir: './dist',
    middleware: {
      0: null
    }
  },
  open: false
};
