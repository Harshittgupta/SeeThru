# Multi-stage: build the SPA with Node, serve the static output with nginx.
# The runtime image contains no Node and no source -- just the built assets.

FROM node:20-alpine AS build
WORKDIR /app
# Copy manifests first so `npm ci` is cached until deps actually change.
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
# nginx runs in the foreground so the container stays up.
CMD ["nginx", "-g", "daemon off;"]
